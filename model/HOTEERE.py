from itertools import combinations
import pdb
from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from transformers import AdamW, get_linear_schedule_with_warmup
from data_modules.input_example import InputExample
from model.generator import GenOT
from model.selector import SentenceSelectOT
from utils.tools import compute_f1


class HOTEERE(pl.LightningModule):
    """
    TODO: sampling from p(Y) (p(y_i) = sum_j(pi_yi_xj))
    """
    def __init__(self,
                weight_source_perserve_ev_reward: float,
                weight_gen_perserve_ev_reward: float,
                weight_sent_diversity_reward: float,
                weight_mle: float,
                num_training_step: int,
                selector_lr: float,
                generator_lr: float,
                weight_selector_loss: float = 0.5,
                OT_eps: float = 0.1,
                OT_max_iter: int = 50,
                OT_reduction: str = 'mean',
                dropout: float = 0.5,
                null_sentence_prob: float = 0.5,
                kg_weight: float = 0.2,
                finetune_selector_encoder: bool = True,
                finetune_in_OT_generator: bool = True,
                encoder_name_or_path: str = 't5-base',
                tokenizer_name_or_path: str = 't5-base',
                n_selected_sents: int = 5,
                null_word_prob: float = 0.5,
                n_selected_words: int = 10,
                output_max_length: int = 32,) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.weight_source_perserve_ev_reward = weight_source_perserve_ev_reward
        self.weight_gen_perserve_ev_reward = weight_gen_perserve_ev_reward
        self.weight_sent_diversity_reward = weight_sent_diversity_reward
        self.weight_mle = weight_mle
        self.weight_selector_loss = weight_selector_loss
        self.num_training_step = num_training_step
        self.selector_lr = selector_lr
        self.generator_lr = generator_lr

        self.hidden_size = 768 if 'base' in encoder_name_or_path else 1024
        self.selector = SentenceSelectOT(hidden_size=self.hidden_size,
                                        OT_eps=OT_eps,
                                        OT_max_iter=OT_max_iter,
                                        OT_reduction=OT_reduction,
                                        dropout=dropout,
                                        null_prob=null_sentence_prob,
                                        kg_weight=kg_weight,
                                        finetune=finetune_selector_encoder,
                                        encoder_name_or_path=encoder_name_or_path,
                                        tokenizer_name_or_path=tokenizer_name_or_path,
                                        n_selected_sents=n_selected_sents)
        self.generator = GenOT(pretrain_model=encoder_name_or_path,
                            tokenizer=tokenizer_name_or_path,
                            finetune_in_OT=finetune_in_OT_generator,
                            OT_eps=OT_eps,
                            OT_max_iter=OT_max_iter,
                            OT_reduction=OT_reduction,
                            null_prob=null_word_prob,
                            n_selected_words=n_selected_words,
                            output_max_length=output_max_length)
        
        self.best_vals = None
        self.model_results = None
    
    def format_input_for_selector(self, batch: List[InputExample]):
        doc_sentences: List[List[str]] = []
        doc_embs: List[torch.Tensor] = [] # List of tensors which has size (ns, hidden_size)
        kg_sents: List[List[str]] = []
        kg_sent_embs: List[torch.Tensor] = []
        host_ids: List[List[int]] = [] # index of host sentence in document
        context_sentences_scores: List[List[int]] = [] # scores is computed by doc-coref tree
        for ex in batch:
            doc_sentences.append(ex.context)
            doc_embs.append(ex.sentences_presentation)
            kg_sents.append(ex.kg_sents)
            kg_sent_embs.append(ex.kg_sents_embs)
            host_sent_id = [trg.sent_id for trg in ex.triggers]
            host_ids.append(host_sent_id)
            context_sentences_scores.append(ex.score_over_doc_tree['score'])
        
        return doc_sentences, doc_embs, kg_sents, kg_sent_embs, host_ids, context_sentences_scores
    
    def format_input_for_generator(self, batch: List[InputExample], selected_senteces: List[List[Tuple[int, str]]]=None):
        contexts: List[str] = []
        head_positions: List[Tuple[int, int]] = [] # char position
        tail_positions: List[Tuple[int, int]] = []
        labels: List[str] = []
        head_sentences: List[str] = []
        head_pos_in_sent: List[Tuple[int, int]] = []
        tail_sentences: List[str] = []
        tail_pos_in_sent: List[Tuple[int, int]] = []
        if selected_senteces == None:
            selected_senteces = []
            for ex in batch:
                selected_senteces.append([(trg.sent_id, ex.context[trg.sent_id]) for trg in ex.triggers])
        for ex, context_sentences in zip(batch, selected_senteces):
            host_sentences = [(trg.sent_id, ex.context[trg.sent_id]) for trg in ex.triggers]
            context_sentences = list(set(host_sentences + context_sentences))
            context_sentences.sort(key=lambda x: x[0])
            context = []
            sent_before_head = []
            sent_before_tail = []
            for s in context_sentences:
                context.append(s[1])
                if s[0] < ex.triggers[0].sent_id:
                    sent_before_head.append(s[1])
                if s[0] < ex.triggers[1].sent_id:
                    sent_before_tail.append(s[1])
            context = ' '.join(context)

            head_sentence = ex.context[ex.relations[0].head.sent_id]
            tail_sentence = ex.context[ex.relations[0].tail.sent_id]
            _head_pos_in_sent = [ex.relations[0].head.start_char_in_sent, ex.relations[0].head.end_char_in_sent]
            _tail_pos_in_sent = [ex.relations[0].tail.start_char_in_sent, ex.relations[0].tail.end_char_in_sent]
            head_sentences.append(head_sentence)
            head_pos_in_sent.append(_head_pos_in_sent)
            tail_sentences.append(tail_sentence)
            tail_pos_in_sent.append(_tail_pos_in_sent)

            contexts.append(context)
            len_before_head_sent = len(' '.join(sent_before_head)) + 1 if len(sent_before_head) != 0 else 0 # space
            len_before_tail_sent = len(' '.join(sent_before_tail)) + 1 if len(sent_before_tail) != 0 else 0 # space
            head_position = (ex.triggers[0].start_char_in_sent + len_before_head_sent, ex.triggers[0].end_char_in_sent + len_before_head_sent)
            tail_position = (ex.triggers[1].start_char_in_sent + len_before_tail_sent, ex.triggers[1].end_char_in_sent + len_before_tail_sent)
            head_positions.append(head_position)
            tail_positions.append(tail_position)
            assert context[head_position[0]: head_position[1]] == head_sentence[_head_pos_in_sent[0]: _head_pos_in_sent[1]], \
            f"{context[head_position[0]: head_position[1]]} == {head_sentence[_head_pos_in_sent[0]: _head_pos_in_sent[1]] - {context}}"
            assert context[tail_position[0]: tail_position[1]] == tail_sentence[_tail_pos_in_sent[0]: _tail_pos_in_sent[1]], \
            f"{context[tail_position[0]: tail_position[1]]} == {tail_sentence[_tail_pos_in_sent[0]: _tail_pos_in_sent[1]]} - {context}"
            labels.append(ex.relations[0].type.natural)

        return contexts, head_positions, tail_positions, labels, head_sentences, head_pos_in_sent, tail_sentences, tail_pos_in_sent

    def training_step(self, batch: List[InputExample], batch_idx) -> STEP_OUTPUT:
        if self.current_epoch > 1:
            doc_sentences, doc_embs, kg_sents, kg_sent_embs, host_ids, context_sentences_scores = self.format_input_for_selector(batch)
            sent_OT_cost, selector_log_probs, selected_sents, sentence_diversity_reward = self.selector(doc_sentences=doc_sentences,
                                                                                                    doc_embs=doc_embs,
                                                                                                    kg_sents=kg_sents,
                                                                                                    kg_sent_embs=kg_sent_embs,
                                                                                                    host_ids=host_ids, 
                                                                                                    context_sentences_scores=context_sentences_scores, 
                                                                                                    is_training=True)

            contexts, head_positions, tail_positions, labels, \
            head_sentences, head_pos_in_sent, tail_sentences, tail_pos_in_sent  = self.format_input_for_generator(batch, selected_sents)
            if batch[0].input_format_type == 'ECI_input':
                task_description_words = ['cause', 'because']
                task = 'ECI'
            word_OT_cost, generator_log_probs, mle_loss, \
            predicted_seq, gold_seqs, performance_reward, \
            generator_preserving_event_reward = self.generator(task=task,
                                                            input_format_type=batch[0].input_format_type,
                                                            output_format_type=batch[0].output_format_type,
                                                            contexts=contexts,
                                                            head_positions=head_positions,
                                                            tail_positions=tail_positions,
                                                            task_description_words=task_description_words,
                                                            labels=labels,
                                                            head_sentences=head_sentences,
                                                            head_pos_in_sent=head_pos_in_sent,
                                                            tail_sentences=tail_sentences,
                                                            tail_pos_in_sent=tail_pos_in_sent,
                                                            is_training=True,
                                                            is_warm_up=False)
            selector_rl_loss = - (self.weight_sent_diversity_reward * sentence_diversity_reward + (1.0 - self.weight_sent_diversity_reward) * performance_reward) * selector_log_probs
            generator_rl_loss = - (self.weight_gen_perserve_ev_reward * generator_preserving_event_reward + (1.0 - self.weight_gen_perserve_ev_reward) * performance_reward) * generator_log_probs
            
            loss = self.weight_selector_loss * selector_rl_loss + \
                    (1.0 - self.weight_selector_loss) * (self.weight_mle * mle_loss + (1.0 - self.weight_mle) * generator_rl_loss)
            self.log_dict({'gen_preseve': generator_preserving_event_reward,
                        'divers': sentence_diversity_reward,
                        'perfromance': performance_reward,
                        's_OT': sent_OT_cost,
                        'w_OT': word_OT_cost,
                        'mle': mle_loss, 
                        }, prog_bar=True)
            self.log_dict({'s_rl': selector_rl_loss, 
                        'g_rl': generator_rl_loss,
                        }, prog_bar=False)
            
        else:
            contexts, head_positions, tail_positions, labels, \
            head_sentences, head_pos_in_sent, tail_sentences, tail_pos_in_sent  = self.format_input_for_generator(batch)
            if batch[0].input_format_type == 'ECI_input':
                task_description_words = ['cause', 'because']
                task = 'ECI'
            word_OT_cost, generator_log_probs, mle_loss, \
            predicted_seq, gold_seqs, performance_reward, \
            generator_preserving_event_reward = self.generator(task=task,
                                                            input_format_type=batch[0].input_format_type,
                                                            output_format_type=batch[0].output_format_type,
                                                            contexts=contexts,
                                                            head_positions=head_positions,
                                                            tail_positions=tail_positions,
                                                            task_description_words=task_description_words,
                                                            labels=labels,
                                                            head_sentences=head_sentences,
                                                            head_pos_in_sent=head_pos_in_sent,
                                                            tail_sentences=tail_sentences,
                                                            tail_pos_in_sent=tail_pos_in_sent,
                                                            is_training=True,
                                                            is_warm_up=True)
            loss = mle_loss
            self.log_dict({'mle': mle_loss}, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        if self.current_epoch > 1:
            doc_sentences, doc_embs, kg_sents, kg_sent_embs, host_ids, context_sentences_scores = self.format_input_for_selector(batch)
            sent_OT_cost, selector_log_probs, selected_sents, sentence_diversity_reward = self.selector(doc_sentences=doc_sentences,
                                                                                                    doc_embs=doc_embs,
                                                                                                    kg_sents=kg_sents,
                                                                                                    kg_sent_embs=kg_sent_embs,
                                                                                                    host_ids=host_ids, 
                                                                                                    context_sentences_scores=context_sentences_scores, 
                                                                                                    is_training=True)

            contexts, head_positions, tail_positions, labels, \
            head_sentences, head_pos_in_sent, tail_sentences, tail_pos_in_sent  = self.format_input_for_generator(batch, selected_sents)
            if batch[0].input_format_type == 'ECI_input':
                task_description_words = ['cause', 'because']
                task = 'ECI'
            word_OT_cost, generator_log_probs, mle_loss, \
            predicted_seq, gold_seqs, performance_reward, \
            generator_preserving_event_reward = self.generator(task=task,
                                                            input_format_type=batch[0].input_format_type,
                                                            output_format_type=batch[0].output_format_type,
                                                            contexts=contexts,
                                                            head_positions=head_positions,
                                                            tail_positions=tail_positions,
                                                            task_description_words=task_description_words,
                                                            labels=labels,
                                                            head_sentences=head_sentences,
                                                            head_pos_in_sent=head_pos_in_sent,
                                                            tail_sentences=tail_sentences,
                                                            tail_pos_in_sent=tail_pos_in_sent,
                                                            is_training=False,
                                                            is_warm_up=False)
            
        else:
            contexts, head_positions, tail_positions, labels, \
            head_sentences, head_pos_in_sent, tail_sentences, tail_pos_in_sent  = self.format_input_for_generator(batch)
            if batch[0].input_format_type == 'ECI_input':
                task_description_words = ['cause', 'because']
                task = 'ECI'
            word_OT_cost, generator_log_probs, mle_loss, \
            predicted_seq, gold_seqs, performance_reward, \
            generator_preserving_event_reward = self.generator(task=task,
                                                            input_format_type=batch[0].input_format_type,
                                                            output_format_type=batch[0].output_format_type,
                                                            contexts=contexts,
                                                            head_positions=head_positions,
                                                            tail_positions=tail_positions,
                                                            task_description_words=task_description_words,
                                                            labels=labels,
                                                            head_sentences=head_sentences,
                                                            head_pos_in_sent=head_pos_in_sent,
                                                            tail_sentences=tail_sentences,
                                                            tail_pos_in_sent=tail_pos_in_sent,
                                                            is_training=False,
                                                            is_warm_up=True)
        
        return predicted_seq, gold_seqs, task

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        gold_seqs = []
        pred_seqs = []
        task = outputs[0][-1]
        for output in outputs:
            pred_seqs.extend(output[0])
            gold_seqs.extend(output[1])
        p, r, f1 = compute_f1(pred_seqs, gold_seqs, task=task)
        self.log('f1_dev', f1, prog_bar=True)
        self.log_dict({'p_dev': p, 'r_dev': r}, prog_bar=False)
        if self.best_vals==None or f1 >= self.best_vals[-1]:
            print(f"\nBetter: {(p, r, f1)}\n")
            self.best_vals = [p, r, f1]
        return f1

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        doc_sentences, doc_embs, kg_sents, kg_sent_embs, host_ids, context_sentences_scores = self.format_input_for_selector(batch)
        sent_OT_cost, selector_log_probs, selected_sents, sentence_diversity_reward = self.selector(doc_sentences=doc_sentences,
                                                                                                    doc_embs=doc_embs,
                                                                                                    kg_sents=kg_sents,
                                                                                                    kg_sent_embs=kg_sent_embs,
                                                                                                    host_ids=host_ids, 
                                                                                                    context_sentences_scores=context_sentences_scores, 
                                                                                                    is_training=False)

        contexts, head_positions, tail_positions, labels, \
        head_sentences, head_pos_in_sent, tail_sentences, tail_pos_in_sent  = self.format_input_for_generator(batch, selected_sents)
        if batch[0].input_format_type == 'ECI_input':
            task_description_words = ['cause', 'because']
            task = 'ECI'
        word_OT_cost, generator_log_probs, mle_loss, \
        predicted_seq, gold_seqs, performance_reward, \
        generator_preserving_event_reward = self.generator(task=task,
                                                        input_format_type=batch[0].input_format_type,
                                                        output_format_type=batch[0].output_format_type,
                                                        contexts=contexts,
                                                        head_positions=head_positions,
                                                        tail_positions=tail_positions,
                                                        task_description_words=task_description_words,
                                                        labels=labels,
                                                        head_sentences=head_sentences,
                                                        head_pos_in_sent=head_pos_in_sent,
                                                        tail_sentences=tail_sentences,
                                                        tail_pos_in_sent=tail_pos_in_sent,
                                                        is_training=False,
                                                        is_warm_up=False)
        
        return predicted_seq, gold_seqs, task

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        gold_seqs = []
        pred_seqs = []
        task = outputs[0][-1]
        for output in outputs:
            pred_seqs.extend(output[0])
            gold_seqs.extend(output[1])
        p, r, f1 = compute_f1(pred_seqs, gold_seqs, task=task)
        self.log_dict({'hp_metric': f1})
        self.log_dict({'hp_metric/p_test': p, 'hp_metric/r_test': r})
        self.model_results = (p, r, f1)
        return f1

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        num_batches = self.num_training_step / self.trainer.accumulate_grad_batches
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_pretrain_parameters = [
            {
                "params": [p for n, p in self.selector.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.05,
                "lr": self.selector_lr
            },
             {
                "params": [p for n, p in self.selector.named_parameters() if  any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.selector_lr
            },
            {
                "params": [p for n, p in self.generator.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.05,
                "lr": self.generator_lr
            },
            {
                "params": [p for n, p in self.generator.named_parameters() if  any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.generator_lr
            },]
        
        optimizer = AdamW(optimizer_grouped_pretrain_parameters, betas=[0.9, 0.999], eps=1e-8)
        num_warmup_steps = 0.1 * num_batches
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval': 'step'
            }
        }
