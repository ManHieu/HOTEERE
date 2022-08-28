from collections import defaultdict
from math import cos
import pdb
from typing import List
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from model.sinkhorn import SinkhornDistance
from transformers import T5Tokenizer, T5EncoderModel


class SentenceSelectOT(nn.Module):
    def __init__(self,
                hidden_size: int,
                OT_eps: float = 0.1,
                OT_max_iter: int = 50,
                OT_reduction: str = 'mean',
                dropout: float = 0.5,
                null_prob: float = 0.5,
                kg_weight: float = 0.2,
                finetune: bool = True,
                encoder_name_or_path: str = 't5-base',
                tokenizer_name_or_path: str = 't5-base',
                n_selected_sents: int = 5,
                use_rnn: bool = False,
                ):
        super().__init__()
        self.finetune = finetune
        if self.finetune == True:
            self.encoder = T5EncoderModel.from_pretrained(encoder_name_or_path, output_hidden_states=True)
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name_or_path)
            self.hidden_size = 768 if 'base' in encoder_name_or_path else 1024
        else:
            self.hidden_size = hidden_size
        self.use_rnn = use_rnn
        self.sinkhorn = SinkhornDistance(eps=OT_eps, max_iter=OT_max_iter, reduction=OT_reduction)
        self.null_prob = null_prob
        self.kg_weight = kg_weight
        self.dropout = nn.Dropout(dropout)
        self.n_selected_sents = n_selected_sents
        self.cos = nn.CosineSimilarity(dim=1)
    
    def encode_with_rnn(self, inp: torch.Tensor(), ls: List[int]) -> torch.Tensor(): # (batch_size, max_ns, hidden_dim*2)
        packed = pack_padded_sequence(inp, ls, batch_first=True, enforce_sorted=False)
        rnn_encode, _ = self.lstm(packed)
        outp, _ = pad_packed_sequence(rnn_encode, batch_first=True)
        return outp

    @torch.no_grad()
    def compute_sentence_diversity_reward(self, host_sent_embs, selected_sent_embs):
        reward = []
        assert len(host_sent_embs) == len(selected_sent_embs)
        for host_emb, selected_emb in zip(host_sent_embs, selected_sent_embs):
            if len(selected_emb) != 0:
                _reward = []
                selected_emb = torch.stack(selected_emb)
                for i in range(host_emb.size(0)):
                    cos_distance = 1 - self.cos(host_emb[i].unsqueeze(0), selected_emb)
                    _reward.append(float(torch.sum(cos_distance)) / cos_distance.size(-1))
                reward.append(sum(_reward) / len(_reward))
            else:
                reward.append(0.1)
        return sum(reward) / len(reward)

    def forward(self,
                doc_sentences: List[List[str]],
                doc_embs: List[torch.Tensor], # List of tensors which has size (ns, hidden_size)
                kg_sents: List[List[str]],
                kg_sent_embs: List[torch.Tensor],
                host_ids: List[List[int]], # index of host sentence in document
                context_sentences_scores: List[List[int]], # scores is computed by doc-coref tree
                is_training: bool = True):
        """
        [x]TODO: Add a trainabel filler which will dot product with embedding of X and Y to get new embedding vetor before feed them into Optimal Transport
        TODO: Softmin instead of softmax (encourage sample farer sentence?)
        TODO: Consider chossing k sentences as k consecutive actions 
        TODO: Filler is a (1, hidden_size) tensor instead of linear
        """
        bs = len(host_ids)
        if self.finetune == True:
            doc_embs = []
            kg_sent_embs = []
            for i in range(bs):
                doc_input_ids = self.tokenizer(['</s>' + sent for sent in doc_sentences[i]], 
                                            return_tensors="pt", 
                                            padding='longest',
                                            truncation=True,
                                            ).input_ids
                doc_outputs = self.encoder(input_ids=doc_input_ids.cuda())
                doc_embs.append(doc_outputs.last_hidden_state[:, 0])

                kg_input_ids = self.tokenizer(['</s>' + sent for sent in kg_sents[i]], 
                                            return_tensors="pt",
                                            padding='longest',
                                            truncation=True,).input_ids
                kg_outputs = self.encoder(input_ids=kg_input_ids.cuda())
                kg_sent_embs.append(kg_outputs.last_hidden_state[:, 0])

        ns = [doc_emb.size(0) for doc_emb in doc_embs]
        n_kg_sent = [kg_sent_emb.size(0) for kg_sent_emb in kg_sent_embs]
        if self.use_rnn == True:
            doc_embs = pad_sequence(doc_embs, batch_first=True)
            doc_embs = self.encode_with_rnn(doc_embs, ns)
        else:
            doc_embs = pad_sequence(doc_embs, batch_first=True)
        kg_sent_embs = pad_sequence(kg_sent_embs, batch_first=True)
        doc_embs = self.dropout(doc_embs)
        kg_sent_embs = self.dropout(kg_sent_embs)
        
        X_presentations = []
        Y_presentations = []
        P_X = []
        P_Y = []
        context_ids = []
        host_embs = []
        for i in range(bs):
            _ns = ns[i]
            _n_kg_sent = n_kg_sent[i]
            host_id = host_ids[i]
            context_id = list(set(range(_ns)) - set(host_id))
            context_ids.append(context_id)
            host_id = list(set(host_id))
            host_sentences_emb = doc_embs[i, host_id]
            host_embs.append(host_sentences_emb)
            context_sentences_emb = doc_embs[i, context_id]
            null_presentation = torch.mean(context_sentences_emb, dim=0).unsqueeze(0)
            X_emb = torch.cat([null_presentation, host_sentences_emb], dim=0)
            X_presentations.append(X_emb)
            context_sentences_emb = torch.cat([context_sentences_emb, kg_sent_embs[i, 0:_n_kg_sent]], dim=0)
            Y_presentations.append(context_sentences_emb)

            X_maginal = torch.tensor([1.0 / len(host_id)] * len(host_id), dtype=torch.float)
            X_maginal = [torch.tensor([self.null_prob]), (1 - self.null_prob) * X_maginal]
            X_maginal = torch.cat(X_maginal, dim=0)
            P_X.append(X_maginal)
            context_score = context_sentences_scores[i]
            Y_maginal = torch.tensor(context_score, dtype=torch.float)
            Y_maginal = F.softmax(Y_maginal) # farer sentence, higher sample rate 
            Y_maginal = torch.cat([Y_maginal * (1.0 - self.kg_weight), self.kg_weight * torch.tensor([1.0 / _n_kg_sent] * _n_kg_sent, dtype=torch.float)], 
                                dim=0)
            P_Y.append(Y_maginal)
            assert Y_maginal.size(0) == context_sentences_emb.size(0)
            assert X_maginal.size(0) == X_emb.size(0)

        X_presentations = pad_sequence(X_presentations, batch_first=True)
        Y_presentations = pad_sequence(Y_presentations, batch_first=True)
        P_X = pad_sequence(P_X, batch_first=True)
        P_Y = pad_sequence(P_Y, batch_first=True)

        cost, pi, C = self.sinkhorn(X_presentations, Y_presentations, P_X, P_Y, cuda=True) # pi: (bs, nX, nY)
        # =============================An action with top-5 opts====================================
        # aligns = []
        # for i in range(bs):
        #     nY = n_kg_sent[i] + len(context_ids[i])
        #     _host_align = torch.sum(pi[i, 1:, :nY], dim=0)
        #     _host_align = torch.softmax(_host_align, dim=0)
        #     aligns.append(_host_align)
        # aligns = pad_sequence(aligns, batch_first=True) # bs x nY
        # aligns = 1.0 - (1.0 - aligns) ** self.n_selected_sents # https://doi.org/10.1145/3289600.3290999
        
        # log_probs = torch.zeros((bs))
        # selected_sents = [[]*bs]
        # for i in range(self.n_selected_sents):
        #     if is_training:
        #         probs = torch.distributions.Categorical(probs=aligns)
        #         selected_senentence = probs.sample()
        #         log_probs = log_probs + probs.log_prob(selected_senentence)
        #     else:
        #         sorted_probs, idxs = torch.sort(aligns, dim=1, descending=True)
        #         selected_senentence = idxs[:, i]
        #         log_probs = log_probs + torch.log(sorted_probs[:, i])
        #     for j in range(bs):
        #         if selected_senentence[j] < len(context_ids[j]) + n_kg_sent[j]:
        #             if selected_senentence[j] < len(context_ids[j]):
        #                 context_sent_id = context_ids[j][selected_senentence[j]]
        #                 context_sent = doc_sentences[j][context_sent_id]
        #                 selected_sents[j].append((context_sent_id, context_sent))
        #             else:
        #                 kg_sent_id = selected_senentence[j] - len(context_ids[j])
        #                 kg_sent = kg_sents[j][kg_sent_id]
        #                 selected_sents[j].append((9999, kg_sent)) # this means we put kg sent in the tail of the augmented doc
        #========================================================================================================
        values, aligns = torch.max(pi, dim=1)
        selected_sents = []
        selected_sent_embs = []
        mask = torch.zeros_like(pi)
        for i in range(bs):
            _selected_sents_with_mapping = {}
            nY = n_kg_sent[i] + len(context_ids[i])
            for j in range(nY):
                if aligns[i, j] != 0:
                    prob = pi[i, aligns[i, j], j]
                    mapping = (i, aligns[i, j], j)
                    if j < len(context_ids[i]):
                        context_sent_id = context_ids[i][j]
                        context_sent = doc_sentences[i][context_sent_id]
                        selected_sent = (context_sent_id, context_sent)
                        selected_sent_emb = doc_embs[i, context_sent_id]
                    else:
                        kg_sent_id = j - len(context_ids[i])
                        kg_sent = kg_sents[i][kg_sent_id]
                        selected_sent = (9999, kg_sent) # this means we put kg sent in the tail of the augmented doc
                        selected_sent_emb = kg_sent_embs[i, kg_sent_id]
                    if _selected_sents_with_mapping.get(j) != None:
                        if _selected_sents_with_mapping[j][0] < prob:
                            _selected_sents_with_mapping[j] = (prob, mapping, j, selected_sent_emb, selected_sent)
                    else:
                        _selected_sents_with_mapping[j] = (prob, mapping, j, selected_sent_emb, selected_sent)
            
            _selected_sents = []
            _selected_sent_embs = []
            if self.n_selected_sents != None:
                sorted_by_prob = list(_selected_sents_with_mapping.values())
                sorted_by_prob.sort(key=lambda x: x[0], reverse=True)
                for item in sorted_by_prob[:self.n_selected_sents]:
                    _selected_sents.append(item[-1])
                    _selected_sent_embs.append(item[-2])
                    indicate = item[1]
                    mask[indicate[0], indicate[1], indicate[2]] = 1
            else:
                for item in _selected_sents_with_mapping.values():
                    _selected_sents.append(item[-1])
                    _selected_sent_embs.append(item[-2])
                    indicate = item[1]
                    mask[indicate[0], indicate[1], indicate[2]] = 1
            selected_sents.append(_selected_sents)
            selected_sent_embs.append(_selected_sent_embs)
        
        log_probs = torch.sum((torch.log(pi + 1e-10) * mask).view((bs, -1)), dim=-1)
        sentence_diversity_reward = self.compute_sentence_diversity_reward(host_sent_embs=host_embs, selected_sent_embs=selected_sent_embs)
        return cost, torch.mean(log_probs, dim=0), selected_sents, sentence_diversity_reward


            
