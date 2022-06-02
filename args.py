from dataclasses import dataclass, field
from optparse import Option
from typing import Optional, List
import os
from torch import OptionalType
from transformers_modified import TrainingArguments

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    rational_layers:Optional[str] = field(
        default="", metadata={"help": "specify which layer to use rationals, 0-11 means all layers"})
    

    add_ln: bool = field(default=False, metadata={"help":"Whether to add layernorm before rational layers"})

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default="./data",
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    academicBERT: bool = field(
        default=False
    )
    pretrained_layers: Optional[str] = field(
        default="", 
        metadata={"help": "loaded from pretrained layers"}

    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )
@dataclass
class CustomTrainingArguments(TrainingArguments):

    approx_func:Optional[str] = field(
        default="gelu", metadata={"help": "specify which layer to use rationals, 0-11 means all layers"})

    save_rational_plots: bool = False
    zero_shot: bool = False
    run_name: str = field(
        default="run1",
        metadata={
            "help": "wandb run name"
        }
    )
    scale_cnt_limit: int = 5
    do_pruning: bool = False
    sparse_mask_prediction: bool = False

    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: "},
    )

    do_mix: bool = False

    total_training_time: Optional[float] = field(
        default=24.0, metadata={"help": "Max hours for pre-training)"}
    )

    freeze_pl: bool = False

    pl_learning_rate: float = 5e-5

    rational_lr:Optional[float] = field(
        default=1e-2, metadata={"help": "specify which layer to use rationals, 0-11 means all layers"})

    optimizer: Optional[str] = field(default="adam", metadata={"help":"whether to use RecAdam"})

    recadam_anneal_w: Optional[float] = field(default=1.0, metadata={"help":"Weight for the annealing function in RecAdam. Default 1.0."})
    recadam_anneal_fun: Optional[str] = field(default='sigmoid', metadata={"help":"the type of annealing function in RecAdam. Default sigmoid"})

    recadam_anneal_k: Optional[float] = field(default=0.5, metadata={"help":"k for the annealing function in RecAdam."})                    
    recadam_anneal_t0: Optional[int] = field(default=250, metadata={"help":"t0 for the annealing function in RecAdam."})
    recadam_pretrain_cof: Optional[float] = field(default=5000.0, metadata={"help":"tCoefficient of the quadratic penalty in RecAdam. Default 5000.0."})

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    task_name: str = field(
    default=None,
    metadata={
        "help": "task name to record. "
    },
    )


    tb_dir: Optional[str] = field(
        default=None,
        metadata={"help": "tensorboard_dir"},
    )

    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})




    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()

        elif self.dataset_name is not None:
            pass

        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")

        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`train_file` should be a csv, a json or a txt file.")
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`validation_file` should be a csv, a json or a txt file.")

@dataclass
class KD_Arguments:
    alpha_ce: Optional[float] = field(default=0.0, metadata={})
    alpha_mlm: Optional[float] = field(default=0.0, metadata={})
    alpha_cos: Optional[float] = field(default=0.0, metadata={})
    alpha_clm: Optional[float] = field(default=0.0, metadata={})
    alpha_mse: Optional[float] = field(default=0.0, metadata={})
    temperature: Optional[float] = field(default=2.0, metadata={"help":"Temperature for the softmax temperature"})


@dataclass
class DeepspeedArguments:
    """
    DeepspeedArguments
    """

    _argument_group_name = "Deepspeed Arguments"

    # deepspeed: Optional[bool] = field(default=False, metadata={"help": "Use deepspeed."})

    deepspeed_transformer_kernel: Optional[bool] = field(
        default=False, metadata={"help": "Use DeepSpeed transformer kernel to accelerate."}
    )

    stochastic_mode: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use stochastic mode for high-performance transformer kernel.",
        },
    )

    attention_dropout_checkpoint: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use DeepSpeed transformer kernel memory optimization to checkpoint dropout output.",
        },
    )

    normalize_invertible: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use DeepSpeed transformer kernel memory optimization to perform invertible normalize backpropagation.",
        },
    )

    gelu_checkpoint: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use DeepSpeed transformer kernel memory optimization to checkpoint GELU activation.",
        },
    )

    gradient_clipping: Optional[float] = field(
        default=1.0, metadata={"help": "Gradient clipping (default 1.0)"}
    )

    steps_per_print: Optional[int] = field(
        default=100, metadata={"help": "Number of steps between training steps print"}
    )

    wall_clock_breakdown: Optional[bool] = field(
        default=False, metadata={"help": "wall clock breakdown"}
    )

    prescale_gradients: Optional[bool] = field(
        default=False, metadata={"help": "Scale gradients before doing allreduce"}
    )

    gradient_predivide_factor: Optional[int] = field(
        default=None, metadata={"help": "gradient predivide factor"}
    )

    def __post_init__(self):
        if self.deepspeed_transformer_kernel:
            remove_cuda_compatibility_for_kernel_compilation()


def remove_cuda_compatibility_for_kernel_compilation():
    if "TORCH_CUDA_ARCH_LIST" in os.environ:
        del os.environ["TORCH_CUDA_ARCH_LIST"]





@dataclass
class SchedulerArgs:
    """
    Scheduler Arguments
    """

    _argument_group_name = "Scheduler Arguments"

    lr_schedule: Optional[List[str]] = field(
        default="time",
        metadata={
            "help": "learning rate scheduler type (step/constant_step/time"
        },
    )

    curve: Optional[List[str]] = field(
        default="linear",
        metadata={
            "help": "curve shape (linear/exp)",
        },
    )

    warmup_proportion_list: Optional[List[float]] = field(default=0.06, metadata={"help": "Warmup proportion"})
    decay_rate: Optional[float] = field(default=0.99, metadata={"help": "Decay rate"})
    decay_step: Optional[int] = field(default=1000, metadata={"help": "Decay step"})
    num_warmup_steps: Optional[int] = field(
        default=1000, metadata={"help": "Number of warmup steps"}
    )