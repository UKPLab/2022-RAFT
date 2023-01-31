#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
from signal import SIG_DFL
import sys
import random
from argparse import Namespace
from itertools import chain
from unicodedata import name
import datasets
from datasets import Dataset, DatasetDict, load_dataset, load_metric, concatenate_datasets, load_from_disk
from schedules import get_scheduler
from timeit import default_timer as get_now
import transformers_modified as transformers
from transformers_modified import (CONFIG_MAPPING, MODEL_FOR_MASKED_LM_MAPPING,
                                   AutoConfig, AutoModelForMaskedLM,
                                   AutoTokenizer,
                                   DataCollatorForLanguageModeling,
                                   HfArgumentParser, Trainer, set_seed, AdamW,
                                   EvalPrediction)

from transformers_modified.modeling import BertLMHeadModel
from transformers_modified.trainer_utils import get_last_checkpoint
from transformers_modified.utils.versions import require_version
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from transformers_modified.integrations import INTEGRATION_TO_CALLBACK
from callbacks import CustomWandbCallback
from args import ModelArguments, CustomTrainingArguments, DataTrainingArguments, DeepspeedArguments, SchedulerArgs
from customtrainer import CustomTrainer
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.17.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt"
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def merge_args(arg_list):
    args = Namespace()
    for cur_args in arg_list:
        for key, value in cur_args.__dict__.items():
            setattr(args, key, value)
    return args


@record
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    import os
    # os.environ["NCCL_DEBUG"] = "INFO"
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments,
         DeepspeedArguments, SchedulerArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, ds_args, schedule_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, ds_args, schedule_args = parser.parse_args_into_dataclasses(
        )

    args = merge_args([model_args, data_args, training_args, ds_args])

    args.schedule_args = schedule_args
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
    # Set the verbosity to info of the Transformers logger (on main process only):
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

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if data_args.load_from_disk:
            train_dir = "/storage/ukp/work/fang/rational_bert/data/wikipedia/train/grouped"
            dev_dir = "/storage/ukp/work/fang/rational_bert/data/wikipedia/dev/grouped"

            train_dataset = concatenate_datasets([Dataset.from_file(os.path.join(train_dir,file)) for file in os.listdir(train_dir)])
            dev_dataset = concatenate_datasets([Dataset.from_file(os.path.join(dev_dir,file)) for file in os.listdir(dev_dir)])          
            raw_datasets = DatasetDict({"train":train_dataset, "validation": dev_dataset})
        else:
            raw_datasets = load_dataset(data_args.dataset_name,
                                        data_args.dataset_config_name,
                                        cache_dir=model_args.cache_dir,
                                        )

        
        if data_args.extra_dataset:
            raw_extra_datasets = load_dataset(data_args.extra_dataset,
                                        cache_dir=model_args.cache_dir)

            raw_datasets['train'] = concatenate_datasets([raw_datasets['train'], raw_extra_datasets['train']])

        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(extension,
                                    data_files=data_files,
                                    cache_dir=model_args.cache_dir)

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        # if model_args.academicBERT:
        #     config = PretrainedConfig.from_pretrained(model_args.config_name)
        # else:
        config = AutoConfig.from_pretrained(model_args.config_name,
                                            **config_kwargs)

    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path,
                                            **config_kwargs)
    else:
        config = CONFIG_MAPPING[training_args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

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
    print("rational_layers", rational_layers)

    config.rational_layers = rational_layers
    config.approx_func = training_args.approx_func
    config.logging_steps = training_args.logging_steps
    config.tb_dir = data_args.tb_dir

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name,
                                                  **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if not training_args.do_mix and model_args.model_name_or_path and not model_args.academicBERT:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif model_args.academicBERT:
        
        if model_args.model_name_or_path:
            logger.info('loading academicBERT from local model')
            model = BertLMHeadModel.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                args=args,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None)
        else:
            logger.info('Training academicBERT from scratch')
            model = BertLMHeadModel(config=config, args=args)

        
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)
        if training_args.do_mix:
            config.rational_layers = ['']
            pretrained_model = AutoModelForMaskedLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            pretrained_weights = pretrained_model.state_dict()
            replaced_layers = model_args.pretrained_layers.split(',')
            pretrained_layers = []
            for layer in replaced_layers:
                if '-' in layer:
                    layers = layer.split('-')
                    pretrained_layers.extend([
                        'layer.' + str(i) + '.'
                        for i in range(int(layers[0]),
                                       int(layers[1]) + 1)
                    ])
                else:
                    pretrained_layers.append(layer)
            pretrained_layers.append('embedding')
            replaced_weights = {
                k: v
                for k, v in pretrained_weights.items()
                if any(ly in k for ly in pretrained_layers)
            }
            for k, v in replaced_weights.items():
                v.requires_grad = False
            print("replaced weights", replaced_weights.keys())
            model_dict = model.state_dict()
            model_dict.update(replaced_weights)
            model.load_state_dict(model_dict)

            if training_args.freeze_pl:
                logger.info("freeze pretrained layers!")
                for k, v in model.named_parameters():
                    if k in replaced_weights:
                        v.requires_grad = False

            # for k, v in model.named_parameters():
            #     if v.requires_grad:
            #         print(k)

    model.resize_token_embeddings(len(tokenizer))
    


    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length,
                             tokenizer.model_max_length)
    if data_args.load_from_disk:
        tokenized_datasets = raw_datasets
    else:
        tokenized_datasets = preprocess_data(raw_datasets,model_args, column_names, args, 
        data_args, text_column_name,tokenizer,max_seq_length,training_args)
                            
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        print('train size', len(train_dataset))

        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(
                range(data_args.max_train_samples))
            print('selected training samples', len(train_dataset))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        print('eval_dataset', len(eval_dataset))
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(
                range(data_args.max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            return logits.argmax(dim=-1)

        metric = load_metric("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics
            labels = labels.reshape(-1)
            mask = labels != -100
            labels = labels[mask]
            # for unsparsed prediction the preds size is [batch_size, seq_length]
            if len(preds.shape) == 2:
                preds = preds.reshape(-1)
                preds = preds[mask]
            return metric.compute(predictions=preds, references=labels)

    # Data collator
    # This one will take care of randomly masking the tokens.
    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
    )

    # set optimizer for rational and others
    rational = ['numerator', 'denominator']
    no_decay = ["bias", "LayerNorm.weight"]
    if training_args.do_mix:
        if training_args.freeze_pl:
            grouped_params = [
                {
                    "params": [
                        v for k, v in model.named_parameters()
                        if any(param in k for param in rational)
                    ],
                    "lr":
                    training_args.rational_lr,
                    "weight_decay":
                    training_args.weight_decay
                },
                {
                    "params": [
                        v for k, v in model.named_parameters()
                        if any('layer.' + l + '.' in k
                               for l in rational_layers) and not any(
                                   param in k for param in rational)
                    ],
                    "lr":
                    training_args.learning_rate,
                    "weight_decay":
                    training_args.weight_decay
                },
            ]

        else:
            grouped_params = [
                {
                    "params": [
                        v for k, v in model.named_parameters()
                        if any(param in k for param in rational)
                    ],
                    "lr":
                    training_args.rational_lr,
                    "weight_decay":
                    training_args.weight_decay
                },
                #random initialised ones
                {
                    "params": [
                        v for k, v in model.named_parameters()
                        if not any(l in k
                                   for l in pretrained_layers) and not any(
                                       param in k for param in rational)
                    ],
                    "lr":
                    training_args.learning_rate,
                    "weight_decay":
                    training_args.weight_decay
                },
                # pretrained ones
                {
                    "params": [
                        v for k, v in model.named_parameters()
                        if any(l in k for l in pretrained_layers)
                    ],
                    "lr":
                    training_args.pl_learning_rate,
                    "weight_decay":
                    training_args.weight_decay
                }
            ]
          

    else:

    
        grouped_params = [{
            "params": [
                v for k, v in model.named_parameters()
                if any(param in k for param in rational)
            ],
            "lr":
            training_args.rational_lr,
            "weight_decay":
            training_args.rational_weight_decay
        }, {
            "params": [
                v for k, v in model.named_parameters()
                if not any(param in k for param in rational) and not any(
                    param in k for param in no_decay)
            ],
            "lr":
            training_args.learning_rate,
            "weight_decay":
            training_args.weight_decay
        }, {
            "params": [
                v for k, v in model.named_parameters()
                if not any(param in k for param in rational) and any(
                    param in k for param in no_decay)
            ],
            "lr":
            training_args.learning_rate,
            "weight_decay":
            0.0
        }]
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(grouped_params,
                                     lr=training_args.learning_rate,
                                     weight_decay=training_args.weight_decay)
    elif args.optimizer == "adamw":
        # args.exp_start_marker = get_now()
        optimizer = AdamW(grouped_params,
                          lr=training_args.learning_rate,
                          weight_decay=training_args.weight_decay,
                          betas=(args.adam_beta1, args.adam_beta2),
                          eps=args.adam_epsilon)
    if model_args.academicBERT:
        scheduler = get_scheduler(args.schedule_args, optimizer, args)
    else:
        scheduler = None
    training_args.warmup_proportion_list = schedule_args.warmup_proportion_list



    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
        # callbacks = [rationalcallback],
        optimizers=(optimizer, scheduler))

    # Training
    if training_args.do_train:
        
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(
                f"Sample {index} of the training set: {train_dataset[index]}.")
            assert len(train_dataset[index]['input_ids']) == len(train_dataset[index]['attention_mask']) == len(train_dataset[index]['token_type_ids']) == len(train_dataset[index]['special_tokens_mask'])
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        logging.info("start training!")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (data_args.max_train_samples
                             if data_args.max_train_samples is not None else
                             len(train_dataset))
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        if data_args.tb_dir:
            model.roberta.close_tb_writer()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
            eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "fill-mask"
    }
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs[
                "dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

def preprocess_data(raw_datasets,model_args, column_names, args,  data_args, text_column_name,tokenizer,max_seq_length,training_args):
    if data_args.line_by_line:
    # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False
        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name]
                if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
        )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.

        def tokenize_function(examples):

            return tokenizer(examples[text_column_name], add_special_tokens=False, return_special_tokens_mask=True)

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on every text in dataset",
                cache_file_names={k:os.path.join(model_args.cache_dir, args.dataset_name, k, 'tokenized', k+"_" + args.dataset_name + args.extra_dataset + model_args.tokenizer_name + "_" + str(data_args.max_seq_length) + "_"+ training_args.model_type+'.arrow') for k in raw_datasets}
            )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples, cls_id, sep_id):
            truncated_seq_length = max_seq_length - 2
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # # customize this part to your needs.
            if total_length >= truncated_seq_length:
                total_length = (total_length // truncated_seq_length) * truncated_seq_length
            
            # # Split by chunks of max_len.
            res_dic = {}
            for k, t in concatenated_examples.items():
                if k =="input_ids":
                    res_dic["input_ids"] = [[cls_id]+ t[i : i + truncated_seq_length]+[sep_id] for i in range(0, total_length, truncated_seq_length)]
                
                elif k == "attention_mask":
                    res_dic["attention_mask"] = [[1]+ t[i : i + truncated_seq_length]+[1] for i in range(0, total_length, truncated_seq_length)]
                
                elif k == "token_type_ids":
                    res_dic["token_type_ids"] = [[0]+ t[i : i + truncated_seq_length]+[0] for i in range(0, total_length, truncated_seq_length)]
                
                elif k == "special_tokens_mask":
                    res_dic["special_tokens_mask"] = [[1]+ t[i : i + truncated_seq_length]+[1] for i in range(0, total_length, truncated_seq_length)]

            return res_dic


        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
        if training_args.model_type == "bert":
            cls_id = tokenizer.convert_tokens_to_ids("[CLS]")
            sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
        elif training_args.model_type == "roberta":
            cls_id = tokenizer.convert_tokens_to_ids("<s>")
            sep_id = tokenizer.convert_tokens_to_ids("</s>")
        with training_args.main_process_first(desc="grouping texts together"):
            tokenized_datasets = tokenized_datasets.map(
                lambda x :group_texts(x, cls_id=cls_id,sep_id=sep_id),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
                cache_file_names={k:os.path.join(model_args.cache_dir, args.dataset_name, 'k','grouped', "grouped_" + k +"_" + args.dataset_name + args.extra_dataset + model_args.tokenizer_name + "_" + str(data_args.max_seq_length) + "_"+ training_args.model_type+'.arrow') for k in tokenized_datasets}
            )
    return tokenized_datasets
if __name__ == "__main__":
    INTEGRATION_TO_CALLBACK["wandb"] = CustomWandbCallback
    main()