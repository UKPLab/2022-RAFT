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

import logging
import os
import random
import sys
import collections
import datasets
import numpy as np
from datasets import load_dataset, load_metric
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import random
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from run_squad import train
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
from transformers_modified.file_utils import is_torch_tpu_available, is_sagemaker_mp_enabled
from transformers_modified.tokenization_utils_base import PreTrainedTokenizerBase
from transformers_modified.data.data_collator import DataCollator
from transformers_modified.modeling_utils import PreTrainedModel
from transformers_modified.deepspeed import deepspeed_init
from transformers_modified.trainer_pt_utils import find_batch_size, nested_concat, nested_truncate, nested_numpify, IterableDatasetShard
from torch import nn
from transformers_modified.trainer_callback import TrainerCallback
from transformers_modified.trainer_utils import get_last_checkpoint, denumpify_detensorize, EvalLoopOutput
from transformers_modified import EarlyStoppingCallback
from transformers_modified.utils import check_min_version
from transformers_modified.utils.versions import require_version
from args import ModelArguments, CustomTrainingArguments, DataTrainingArguments, SchedulerArgs
from callbacks import CustomWandbCallback
from transformers_modified.integrations import INTEGRATION_TO_CALLBACK
from pruning_utils import parameters_to_prune, original_params
import torch.nn.utils.prune as prune
from transformers_modified.trainer_utils import PREFIX_CHECKPOINT_DIR, HPSearchBackend
from transformers_modified.modeling import BertForSequenceClassification
from schedules import get_scheduler
import pickle
from rational.torch import Rational
from random_mask import set_uniform_mask

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
                 origin_params: Dict = None,
                 masks: Optional[Dict] = None
                 ):

        super().__init__(model, args, data_collator, train_dataset,
                         eval_dataset, tokenizer, model_init, compute_metrics,
                         callbacks, optimizers)
        self.pruning_step = 0
        self.origin_params = origin_params
        self.masks = masks

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        is_break = False
        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.autocast_smart_context_manager():
            # start = time.time()
            loss, output = self.compute_loss(model, inputs, return_outputs=True)
            # end = time.time()
            # print('compute loss', end-start)
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.fp16:
            if self.state.global_step > 650 and self.args.deepspeed and hasattr(self.optimizer, 'cur_scale'):
                print('optmizer curscale', self.optimizer.cur_scale)
            if (self.args.fp16 or self.args.bf16) and hasattr(self.optimizer, 'cur_scale') and self.optimizer.cur_scale == 1:
                self.scale_counter_at_1 += 1
                logger.info(f"Optimizer scale reach one {self.scale_counter_at_1} times")

            if self.scale_counter_at_1 >= self.args.scale_cnt_limit:
                self.scale_counter_at_1 = 0
                is_break =True
                return loss, is_break
                        
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        if self.masks:
            # print(self.masks)
            for k, v in model.named_parameters():
                if v.grad is not None and 'classifier' not in k:
                    v.grad[~self.masks[k]] = 0

        return loss.detach(), is_break

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
            # print('preds',preds)
            # print('labels', inputs['labels'])
            acc = ((preds.argmax(axis=-1) == inputs["labels"]).type(
                torch.float).mean().item())
            # if self.state.global_step % self.args.logging_steps == 0 and not self.control.should_evaluate:
            #     # logger.info(f"global step {self.state.global_step}")
            #     # logger.info(f"logging step {self.args.logging_steps}")
            #     self.log({"train/accuarcy": acc})

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
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        hidden_states_host = []
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        inter_outputs = None
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            if len(logits.size()) == 3:
                logits = torch.argmax(logits, axis = -1)
            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            # hidden_states_host.append(hidden_states)
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)


            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
            
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        # print("AAAAAAA")
        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)
        # print('eeee')
        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments, SchedulerArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, scheduler_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args,scheduler_args = parser.parse_args_into_dataclasses(
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
            label_list.remove(-1)
            num_labels = len(label_list)
            print("label list", label_list)

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

    elif model_args.academicBERT:
        model = BertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
       
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    print('rational functions', Rational.list)
    if training_args.save_rational_functions:
        file_path = os.path.join('./pkl_files', 'full_data_' + data_args.task_name + '_' + str(training_args.seed)+ '.pkl')
        # file_path = os.path.join('./pkl_files',  'pretrained_' + str(training_args.seed)+ '.pkl')
        with open(file_path,'wb') as f:
            pickle.dump(Rational.list,f)
        exit()

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

    

    # used for subset bitfit
    masks = None
    if training_args.frozen_rf:
        
        for k, v in model.named_parameters():
            if any(name in k for name in rational):
                v.requires_grad=False
        logger.info("rationals have been frozen!")
    
    elif training_args.bitfit or training_args.sub_bias:
        # trained_weights = rational + ["classifier"]
        trained_weights = ["bias","classifier"]
        #bitfit+rational
        # trained_weights += rational
        for k, v in model.named_parameters():
            # print(k)
            if not any(name in k for name in trained_weights):
                v.requires_grad=False
            else:
                print(k)
        logger.info("using bitfit!")
        if training_args.sub_bias:
            num_tune_params = 117
            masks = set_uniform_mask(model, num_tune_params)

    elif training_args.bitfit_rf or training_args.ch_rf:
        if training_args.bitfit_rf:
            trained_weights = rational + ["bias","classifier"]
        else:
            trained_weights = rational + ["classifier"]
        for k, v in model.named_parameters():
            # print(k)
            if not any(name in k for name in trained_weights):
                v.requires_grad=False
            else:
                print(k)
        logger.info("using bitfit(ch)-rf!")

    
    

    grouped_params = [{
        "params": [
            v for k, v in model.named_parameters()
            if any(param in k for param in rational)
        ],
        "lr":
        training_args.rational_lr,
        "weight_decay":training_args.weight_decay
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

    optimizer = AdamW(grouped_params, lr=training_args.learning_rate, weight_decay=training_args.weight_decay)

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
    print('max length',max_seq_length)
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

    train_eval_dataset = raw_datasets["train"]
    train_eval_dataset = train_eval_dataset.shuffle(seed=96)
    train_nums = int(len(train_eval_dataset) * 0.75)
    
    if training_args.do_train or training_args.do_predict:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        
        train_dataset = train_eval_dataset.select(range(train_nums))

        if data_args.max_train_samples is not None:
            random.seed(training_args.seed)
            train_ixs = random.sample(range(len(train_dataset['idx'])), data_args.max_train_samples)
            train_dataset = train_dataset.select(train_ixs)

    if training_args.do_eval:
        # use 25% training data as validation data
        eval_dataset = train_eval_dataset.select(
                range(train_nums, len(train_eval_dataset)))

        if data_args.max_eval_samples is not None and data_args.max_eval_samples < len(eval_dataset):
            random.seed(training_args.seed)
            eval_ixs = random.sample(range(len(eval_dataset['idx'])), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(eval_ixs)

        # else:
        #     eval_dataset = raw_datasets["validation_matched" if data_args.
        #                                 task_name == "mnli" else "validation"]

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:

        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        if training_args.zero_shot:
            predict_dataset = raw_datasets["test"]
            predict_dataset.remove_columns(['premise', 'hypothesis'])
            predict_dataset = predict_dataset.filter(lambda example: example['label']!=-1)
            
        else:
            predict_dataset = raw_datasets["validation_matched" if data_args.
                                        task_name == "mnli" else "validation"]



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
        origin_weights = original_params(model, training_args.model_type)
        callback = None
    else:
        origin_weights = None
        callback = EarlyStoppingCallback(early_stopping_patience=training_args.patience)
    
 
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
        masks=masks,
        callbacks=[callback])


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


        # extract_embedding(training_args, data_args, trainer, label_list, predict_train_outputs.predictions, train_dataset, predict_train_outputs.label_ids, train_hidden_states, predict_train_file=True)

        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["validation_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.

            predict_outputs = trainer.predict(predict_dataset,
                                              metric_key_prefix="test")

            # hidden_states = hidden_states.cpu().detach().numpy()

            metrics = predict_outputs.metrics
            trainer.log_metrics("test", metrics)
            trainer.log(metrics)
            # Rational.export_evolution_graphs(path=os.path.join('./logs/images/input_dist',training_args.run_name))

            # predictions = predict_outputs.predictions
            # true_labels = predict_outputs.label_ids
            # print('true_labels', true_labels)
            # print(label)
            # predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            # output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}_{training_args.name_suffix}.txt")
            # with open(output_predict_file, "w") as writer:
            #     logger.info(f"***** Predict results {task} *****")
            #     # if task_to_keys[data_args.task_name][1] is not None:
            #     #     writer.write("index$#$prediction$#$true$#$example1$#$example2$#$embedding\n")
            #     # else:
            #     writer.write("index$#$prediction$#$true$#$embedding\n")
                
            #     for index, item in enumerate(predictions):
            #         if is_regression:
            #             writer.write(f"{index}\t{item:3.3f}\n")
            #         else:
            #             predict_label = label_list[item]
            #             true_label = label_list[true_labels[index]]
            #             # for sent in task_to_keys[data_args.task_name]:
            #             #     if sent is not None:
            #             #         example = predict_dataset[sent][index]
            #             #         examples.append(example)

            #             # examples = predict_dataset['sentence2'][index]
            #             embedding = " ".join(map(str, hidden_states[index]))
            #             # task_examples = "$#$".join(examples)
            #             # writer2.write(embedding \n)
            #             # example2 = predict_dataset['hypothesis'][index]
            #             writer.write(f"{index}$#${predict_label}$#${true_label}$#${str(embedding)}\n")

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

def extract_embedding(training_args, data_args, trainer, label_list, predictions, predict_dataset, true_labels, hidden_states, predict_train_file=False, task=None):
    if predict_train_file:
        output_predict_folder = os.path.join(training_args.output_dir, 'train', training_args.name_suffix + f"_{task}")
    else:
        output_predict_folder = os.path.join(training_args.output_dir, 'test', training_args.name_suffix +f"_{task}")
    
    if not os.path.exists(output_predict_folder):
        os.makedirs(output_predict_folder)

    embedding_file = os.path.join(output_predict_folder, 'embeddings.txt')
    entities_file = os.path.join(output_predict_folder, 'entities.txt')
    label_file = os.path.join(training_args.output_dir, 'labels.txt')

    if trainer.is_world_process_zero():
        embed_writer = open(embedding_file,'w') 
        entities_writer = open(entities_file,'w')
        with open(label_file, 'w') as f:
            for label in label_list:
                f.write(label+"\n")

        for index, item in enumerate(predictions):

            true_label = label_list[true_labels[index]]
            entities_writer.write(f"{index}\t{true_label}\n")
            embedding = " ".join(map(str, hidden_states[index]))
            embed_writer.write(f"{embedding}\n")
            # if predict_train_file and index >= 3000:
            #     break
            # elif (not predict_train_file) and index >=1000:
            #     break
    entities_writer.close()
    embed_writer.close()


if __name__ == "__main__":
    INTEGRATION_TO_CALLBACK["wandb"] = CustomWandbCallback
    main()
