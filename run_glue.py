#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import imp
import logging
import os
import random
import sys

import datasets
import numpy as np
from datasets import load_dataset, load_metric
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
import transformers_modified as transformers
from transformers_modified import (
    AutoConfig,
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers_modified.file_utils import is_torch_tpu_available
from transformers_modified.tokenization_utils_base import PreTrainedTokenizerBase
from transformers_modified.data.data_collator import DataCollator
from transformers_modified.modeling_utils import PreTrainedModel
from torch import nn
from transformers_modified.trainer_callback import TrainerCallback
from transformers_modified.trainer_utils import get_last_checkpoint
from transformers_modified import EarlyStoppingCallback
from transformers_modified.utils import check_min_version
from transformers_modified.utils.versions import require_version
from args import ModelArguments, CustomTrainingArguments, DataTrainingArguments
from callbacks import CustomWandbCallback
from transformers_modified.integrations import INTEGRATION_TO_CALLBACK
from pruning_utils import parameters_to_prune, original_params
import torch.nn.utils.prune as prune
from transformers_modified.trainer_utils import PREFIX_CHECKPOINT_DIR, HPSearchBackend
from RecAdam import RecAdam, anneal_function

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.18.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt"
)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)

# @dataclass
# class DataTrainingArguments:
#     """
#     Arguments pertaining to what data we are going to input our model for training and eval.
#     Using `HfArgumentParser` we can turn this class
#     into argparse arguments to be able to specify them on
#     the command line.
#     """

#     task_name: Optional[str] = field(
#         default=None,
#         metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
#     )
#     dataset_name: Optional[str] = field(
#         default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
#     )
#     dataset_config_name: Optional[str] = field(
#         default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
#     )
#     max_seq_length: int = field(
#         default=128,
#         metadata={
#             "help": "The maximum total input sequence length after tokenization. Sequences longer "
#             "than this will be truncated, sequences shorter will be padded."
#         },
#     )
#     overwrite_cache: bool = field(
#         default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
#     )
#     pad_to_max_length: bool = field(
#         default=True,
#         metadata={
#             "help": "Whether to pad all samples to `max_seq_length`. "
#             "If False, will pad the samples dynamically when batching to the maximum length in the batch."
#         },
#     )
#     max_train_samples: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
#             "value if set."
#         },
#     )
#     max_eval_samples: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
#             "value if set."
#         },
#     )
#     max_predict_samples: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
#             "value if set."
#         },
#     )
#     train_file: Optional[str] = field(
#         default=None, metadata={"help": "A csv or a json file containing the training data."}
#     )
#     validation_file: Optional[str] = field(
#         default=None, metadata={"help": "A csv or a json file containing the validation data."}
#     )
#     test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

#     def __post_init__(self):
#         if self.task_name is not None:
#             self.task_name = self.task_name.lower()
#             if self.task_name not in task_to_keys.keys():
#                 raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
#         elif self.dataset_name is not None:
#             pass
#         elif self.train_file is None or self.validation_file is None:
#             raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
#         else:
#             train_extension = self.train_file.split(".")[-1]
#             assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
#             validation_extension = self.validation_file.split(".")[-1]
#             assert (
#                 validation_extension == train_extension
#             ), "`validation_file` should have the same extension (csv or json) as `train_file`."

# @dataclass
# class ModelArguments:
#     """
#     Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
#     """

#     model_name_or_path: str = field(
#         metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
#     )
#     config_name: Optional[str] = field(
#         default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
#     )
#     tokenizer_name: Optional[str] = field(
#         default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
#     )
#     cache_dir: Optional[str] = field(
#         default=None,
#         metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
#     )
#     use_fast_tokenizer: bool = field(
#         default=True,
#         metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
#     )
#     model_revision: str = field(
#         default="main",
#         metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
#     )
#     use_auth_token: bool = field(
#         default=False,
#         metadata={
#             "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
#             "with private models)."
#         },
#     )
# @dataclass
# class CustomTrainingArguments(TrainingArguments):
#     approx_func:Optional[str] = field(
#         default="gelu", metadata={"help": "specify which layer to use rationals, 0-11 means all layers"})

#     save_rational_plots: bool = False

#     run_name: str = field(
#         default="run1",
#         metadata={
#             "help": "wandb run name"
#         }
#     )


class CustomTrainer(Trainer):

    def __init__(self,
                 model: Union[PreTrainedModel, nn.Module] = None,
                 args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None,
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 model_init: Callable[[], PreTrainedModel] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction],
                                                    Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer,
                                   torch.optim.lr_scheduler.LambdaLR] = (None,
                                                                         None),
                 origin_params: Dict = None):

        super().__init__(model, args, data_collator, train_dataset,
                         eval_dataset, tokenizer, model_init, compute_metrics,
                         callbacks, optimizers)
        self.pruning_step = 0
        self.origin_params = origin_params

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch,
                                 ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar /
                (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            logger.info('start saving...')
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control)
            if self.args.do_pruning:
                logger.info('start pruning...')

                logger.info(f'pruning {self.pruning_step+1}0%')
                params = parameters_to_prune(model, self.args.model_type)

                prune.global_unstructured(params,
                                          pruning_method=prune.L1Unstructured,
                                          amount=1 / (10 - self.pruning_step))

                self.pruning_step += 1
                if self.pruning_step == 9:
                    self.control.should_training_stop = True
                logger.info('rewinding...')
                model_dict = model.state_dict()
                model_dict.update(self.origin_params)
                model.load_state_dict(model_dict)

                rational = ['numerator', 'denominator']
                grouped_params = [{
                    "params": [
                        v for k, v in model.named_parameters()
                        if any(param in k for param in rational)
                    ],
                    "lr":
                    self.args.rational_lr
                }, {
                    "params": [
                        v for k, v in model.named_parameters()
                        if not any(param in k for param in rational)
                    ],
                    "lr":
                    self.args.learning_rate
                }]
                self.optimizer = torch.optim.Adam(grouped_params,
                                                  lr=self.args.learning_rate)
                self.lr_scheduler = None
                self.lr_scheduler = self.create_scheduler(
                    num_training_steps=self.args.save_steps,
                    optimizer=self.optimizer)

                mask_dict = {}
                for key in model_dict.keys():
                    if 'mask' in key:
                        mask_dict[key] = model_dict[key]
                self._save_mask_dict(mask_dict=mask_dict, trial=None)

    def _save_mask_dict(self, trial, mask_dict):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is not None and trial is not None:
            if self.hp_search_backend == HPSearchBackend.OPTUNA:
                run_id = trial.number
            elif self.hp_search_backend == HPSearchBackend.RAY:
                from ray import tune

                run_id = tune.get_trial_id()
            elif self.hp_search_backend == HPSearchBackend.SIGOPT:
                run_id = trial.id
            run_name = self.hp_name(
                trial) if self.hp_name is not None else f"run-{run_id}"
            run_dir = os.path.join(self.args.output_dir, run_name)
        else:
            run_dir = self.args.output_dir
            self.store_flos()

        output_dir = os.path.join(run_dir, checkpoint_folder)
        torch.save(mask_dict, os.path.join(output_dir, "mask.pt"))

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        if "labels" in inputs:
            preds = outputs.logits.detach()
            acc = ((preds.argmax(axis=-1) == inputs["labels"]).type(
                torch.float).mean().item())
            if self.state.global_step % self.args.logging_steps == 0 and not self.control.should_evaluate:
                # logger.info(f"global step {self.state.global_step}")
                # logger.info(f"logging step {self.args.logging_steps}")
                self.log({"train/accuarcy": acc})

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        +
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir
    ) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(
                training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue",
                                    data_args.task_name,
                                    cache_dir=model_args.cache_dir)
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(data_args.dataset_name,
                                    data_args.dataset_config_name,
                                    cache_dir=model_args.cache_dir)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {
            "train": data_args.train_file,
            "validation": data_args.validation_file
        }

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError(
                    "Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset("csv",
                                        data_files=data_files,
                                        cache_dir=model_args.cache_dir)
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset("json",
                                        data_files=data_files,
                                        cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in [
            "float32", "float64"
        ]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    replaced_layers = model_args.rational_layers.split(',')
    rational_layers = []
    for layer in replaced_layers:
        if '-' in layer:
            layers = layer.split('-')
            rational_layers.extend(
                [str(i) for i in range(int(layers[0]),
                                       int(layers[1]) + 1)])
        else:
            rational_layers.append(layer)

    config.rational_layers = rational_layers
    config.approx_func = training_args.approx_func
    config.add_ln = model_args.add_ln
    config.logging_steps = training_args.logging_steps
    config.tb_dir = data_args.tb_dir

    if training_args.optimizer.lower() == 'recadam' or training_args.do_mix:

        # initialise from scratch
        model = AutoModelForSequenceClassification.from_config(config)
        #
        config.rational_layers = ['']
        config.add_ln = False
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        pretrained_model.to(training_args.device)
        if training_args.do_mix:
            pretrained_weights = pretrained_model.state_dict()
            replaced_layers = model_args.pretrained_layers.split(',')
            pretrained_layers = []
            for layer in replaced_layers:
                if '-' in layer:
                    layers = layer.split('-')
                    pretrained_layers.extend([str(i) for i in range(int(layers[0]), int(layers[1])+1)])
                else:
                    pretrained_layers.append(layer)
            pretrained_layers.append('embedding')
            replaced_weights = {k:v for k, v in pretrained_weights.items() for ly in pretrained_layers if ly in k}
            model_dict = model.state_dict()
            model_dict.update(replaced_weights)
            model.load_state_dict(model_dict)
            # add rational for random initialised model

    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [
            name for name in raw_datasets["train"].column_names
            if name != "label"
        ]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # set optimizer for rational and others
    rational = ["numerator", "denominator"]
    no_decay = ["bias", "LayerNorm.weight"]
    if training_args.optimizer.lower() != "recadam":
        grouped_params = [{
            "params": [
                v for k, v in model.named_parameters()
                if any(param in k for param in rational)
            ],
            "lr":
            training_args.rational_lr,
            "weight_decay":
            training_args.weight_decay
        }, {
            "params": [
                v for k, v in model.named_parameters()
                if not any(param in k for param in rational) and not any(
                    n in k for n in no_decay)
            ],
            "lr":
            training_args.learning_rate,
            "weight_decay":
            training_args.weight_decay
        }, {
            "params": [
                v for k, v in model.named_parameters()
                if not any(param in k
                           for param in rational) and any(n in k
                                                          for n in no_decay)
            ],
            "lr":
            training_args.learning_rate,
            "weight_decay":
            0.0
        }]
        # optimizer = torch.optim.Adam(grouped_params,
                                    #  lr=training_args.learning_rate)
        optimizer = AdamW(grouped_params, lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
    else:
        logger.info("Using RecAdam....")
        grouped_params = [
            {  # rational
                "params": [
                    v for k, v in model.named_parameters()
                    if any(param in k for param in rational)
                ],
                "lr":
                training_args.rational_lr,
                "weight_decay":
                training_args.weight_decay,
                "anneal_w":
                0.0,
                "pretrain_params": [
                    v for k, v in model.named_parameters()
                    if any(param in k for param in rational)
                ]
            },
            {  # decay classification layer
                "params": [
                    v for k, v in model.named_parameters()
                    if not any(param in k for param in rational) and not any(
                        n in k for n in no_decay)
                    and training_args.model_type not in k
                ],
                "lr":
                training_args.learning_rate,
                "weight_decay":
                training_args.weight_decay,
                "anneal_w":
                0.0,
                "pretrain_params": [
                    v for k, v in pretrained_model.named_parameters()
                    if not any(param in k for param in rational) and not any(
                        n in k for n in no_decay)
                    and training_args.model_type not in k
                ]
            },
            {  # no decay classification layer
                "params": [
                    v for k, v in model.named_parameters()
                    if not any(param in k for param in rational) and any(
                        n in k for n in no_decay)
                    and training_args.model_type not in k
                ],
                "lr":
                training_args.learning_rate,
                "weight_decay":
                0.0,
                "anneal_w":
                0.0,
                "pretrain_params": [
                    v for k, v in pretrained_model.named_parameters()
                    if not any(param in k for param in rational) and any(
                        n in k for n in no_decay)
                    and training_args.model_type not in k
                ]
            },
            {  #decay in layer
                "params": [
                    v for k, v in model.named_parameters()
                    if not any(param in k for param in rational) and not any(
                        n in k
                        for n in no_decay) and training_args.model_type in k
                    and "intermediate.LayerNorm" not in k
                ],
                "lr":
                training_args.learning_rate,
                "weight_decay":
                training_args.weight_decay,
                "anneal_w":
                training_args.recadam_anneal_w,
                "pretrain_params": [
                    v for k, v in pretrained_model.named_parameters()
                    if not any(param in k for param in rational) and not any(
                        n in k
                        for n in no_decay) and training_args.model_type in k
                ]
            },

            # no deacy in layer
            {
                "params": [
                    v for k, v in model.named_parameters()
                    if not any(param in k for param in rational) and any(
                        n in k
                        for n in no_decay) and training_args.model_type in k
                    and "intermediate.LayerNorm" not in k
                ],
                "lr":
                training_args.learning_rate,
                "weight_decay":
                0.0,
                "anneal_w":
                training_args.recadam_anneal_w,
                "pretrain_params": [
                    v for k, v in pretrained_model.named_parameters()
                    if not any(param in k for param in rational) and any(
                        n in k
                        for n in no_decay) and training_args.model_type in k
                ]
            },
            {  # layernorm
                "params": [
                    v for k, v in model.named_parameters()
                    if "intermediate.LayerNorm" in k
                ],
                "lr":
                training_args.learning_rate,
                "weight_decay":
                0.0,
                "anneal_w":
                0.0,
                "pretrain_params": [
                    v for k, v in model.named_parameters()
                    if "intermediate.LayerNorm" in k
                ]
            }
        ]

        optimizer = RecAdam(grouped_params,
                            lr=training_args.learning_rate,
                            eps=training_args.adam_epsilon,
                            anneal_fun=training_args.recadam_anneal_fun,
                            anneal_k=training_args.recadam_anneal_k,
                            anneal_t0=training_args.recadam_anneal_t0,
                            pretrain_cof=training_args.recadam_pretrain_cof)
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (model.config.label2id !=
            PretrainedConfig(num_labels=num_labels).label2id
            and data_args.task_name is not None and not is_regression):
        # Some have all caps in their config, some don't.
        label_name_to_id = {
            k.lower(): v
            for k, v in model.config.label2id.items()
        }
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]])
                for i in range(num_labels)
            }
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {
            id: label
            for label, id in config.label2id.items()
        }
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {
            id: label
            for label, id in config.label2id.items()
        }

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = ((examples[sentence1_key], ) if sentence2_key is None else
                (examples[sentence1_key], examples[sentence2_key]))
        result = tokenizer(*args,
                           padding=padding,
                           max_length=max_seq_length,
                           truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1)
                               for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_nums = int(len(train_dataset) * 0.75)
            train_dataset = train_dataset.select(range(train_nums))

    if training_args.do_eval:
        # use 25% training data as validation data
        if data_args.max_train_samples is not None:
            eval_dataset = raw_datasets["train"].select(
                range(train_nums, len(raw_datasets["train"])))
        else:
            eval_dataset = raw_datasets["validation_matched" if data_args.
                                        task_name == "mnli" else "validation"]

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:

        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        predict_dataset = raw_datasets["validation_matched" if data_args.
                                       task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            predict_dataset = predict_dataset.select(
                range(data_args.max_eval_samples))

        # if "test" not in raw_datasets and "test_matched" not in raw_datasets:
        #     raise ValueError("--do_predict requires a test dataset")
        # predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        # if data_args.max_predict_samples is not None:
        #     predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(
                f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        metric = load_metric("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions,
                                               tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds,
                                                                  axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(
                    result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids)**2).mean().item()}
        else:
            return {
                "accuracy":
                (preds == p.label_ids).astype(np.float32).mean().item()
            }

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer,
                                                pad_to_multiple_of=8)
    else:
        data_collator = None
    if training_args.do_pruning:
        origin_weights = original_params(model)
        callback = None
    else:
        origin_weights = None
        callback = EarlyStoppingCallback(early_stopping_patience=3)
    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, None),
        origin_params=origin_weights,
        callbacks=None)
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset if training_args.do_train else None,
    #     eval_dataset=eval_dataset if training_args.do_eval else None,
    #     compute_metrics=compute_metrics,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    # )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (data_args.max_train_samples
                             if data_args.max_train_samples is not None else
                             len(train_dataset))
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        # if data_args.task_name == "mnli":
            # tasks.append("mnli-mm")
            # eval_datasets.append(raw_datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (data_args.max_eval_samples
                                if data_args.max_eval_samples is not None else
                                len(eval_dataset))
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["validation_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            # predict_dataset = predict_dataset.remove_columns("label")
            predict_outputs = trainer.predict(predict_dataset,
                                              metric_key_prefix="test")
            metrics = predict_outputs.metrics
            trainer.log_metrics("test", metrics)
            trainer.log(metrics)

            # predictions = predict_outputs.predictions
            # predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            # output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            # if trainer.is_world_process_zero():
            #     with open(output_predict_file, "w") as writer:
            #         logger.info(f"***** Predict results {task} *****")
            #         writer.write("index\tprediction\n")
            #         for index, item in enumerate(predictions):
            #             if is_regression:
            #                 writer.write(f"{index}\t{item:3.3f}\n")
            #             else:
            #                 item = label_list[item]
            #                 writer.write(f"{index}\t{item}\n")

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-classification"
    }
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    INTEGRATION_TO_CALLBACK["wandb"] = CustomWandbCallback
    main()
