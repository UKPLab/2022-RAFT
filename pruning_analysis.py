from transformers_modified import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from args import ModelArguments, CustomTrainingArguments, DataTrainingArguments
from torch import nn
import os
import torch
import torch.nn.utils.prune as prune
from pruning_utils import parameters_to_prune, get_zero_rate
import seaborn as sns
import matplotlib.pyplot as plt
import os
from transformers_modified.modeling import BertForSequenceClassification

def load_pruned_model(args, task_name, act_func, model, model_dir):
    model_path =os.path.join(model_dir, 'pytorch_model.bin')
    pruned_weights = torch.load(model_path)

    params = parameters_to_prune(model, args.model_type)
    prune.global_unstructured(
        params,
        pruning_method=prune.Identity)
    model_dict = model.state_dict()
    model_dict.update(pruned_weights)
    model.load_state_dict(model_dict)
    pruned_rate, pruned_details = get_zero_rate(model, args.model_type)
    print(pruned_details)
    plt.xticks(rotation=60)
    sns_plot = sns.heatmap(pruned_details, annot=True)
    fig = sns_plot.get_figure()
    image_dir = os.path.join('./logs/images/pruning_analysis/', task_name, act_func)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    fig.savefig(os.path.join(image_dir, f'{pruned_rate}.png'))

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name else model_args.model_name_or_path,
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
    if model_args.academicBERT:
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

    if model_args.rational_layers == "0-11":
        act_func = "rational"
    elif model_args.rational_layers == "" :
        act_func = "original"
    else:
        act_func= "mix"

    load_pruned_model(training_args, data_args.task_name, act_func, model, model_args.model_name_or_path)


if __name__ == '__main__':
    main()