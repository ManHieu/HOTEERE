from dataclasses import dataclass, field
from typing import List, Optional
import transformers


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    datasets: Optional[str] = field(
        default=None,
        metadata={"help": "Comma separated list of dataset names, for training."}
    )

    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to data directory"}
    )

    batch_size: int = field(
        default = 8,
        metadata= {"help": "Batch size"}
    )
    
    n_fold: int = field(
        default = 1,
        metadata={"help": "Number folds of dataset"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Arguments for the Trainer.
    """
    output_dir: str = field(
        default='experiments',
        metadata={"help": "The output directory where the results and model weights will be written."}
    )

    selector_lr: float = field(
        default=5e-5,
        metadata={"help": "Learning rate of selector"}
    )

    generator_lr: float = field(
        default=5e-5,
        metadata={"help": "Learning rate of generator"}
    )

    gradient_clip_val: float = field(
        default=0.0,
        metadata={"help":"Gradient clipping value"}
    )

    num_epoches : int = field(
        default=5,
        metadata={"help": "number pretrain epoches"}
    )

    seed: int = field(
        default=1741,
        metadata={"help": "seeding for reproductivity"}
    )

    weight_source_perserve_ev_reward: float = field(
        default=0.1,
        metadata={"help": "weight of preserving source event's meaning reward"}
    )

    weight_gen_perserve_ev_reward: float = field(
        default=0.1,
        metadata={"help": "weight of preserving generated event's meaning reward"}
    )

    weight_sent_diversity_reward: float = field(
        default=0.1,
        metadata={"help": "weight of the sentence's diversity reward"}
    )

    weight_mle: float = field(
        default=0.8,
        metadata={"help": "weight of generating mle loss"}
    )

    weight_selector_loss: float = field(
        default=0.5,
        metadata={"help": "weight of selector loss"}
    )

    finetune_selector_encoder: bool = field(
        default=True,
        metadata={"help": "Fine-tune selector encoder or not"}
    )

    finetune_in_OT_generator: bool = field(
        default=True,
        metadata={"help": "Fine-tune generator encoder (in OT) or not"}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    null_sentence_prob: float = field(
        default=0.5,
        metadata={"help": "Probability of null senentence in selector"}
    )

    kg_weight: float = field(
        default=0.5,
        metadata={"help": "Probability of KG senentence in selector"}
    )

    n_selected_sents: Optional[int] = field(
        default=None,
        metadata={"help": "Number selected sentences"}
    )

    null_word_prob: float = field(
        default=0.5,
        metadata={"help": "Probability of null word in generator"}
    )

    n_selected_words: Optional[int] = field(
        default=None,
        metadata={"help": "Number selected words"}
    )

    output_max_length: Optional[int] = field(
        default=64,
        metadata={"help": "Max length of Output sequences"}
    )



