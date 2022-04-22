from email.policy import default
import torch
from collections import defaultdict
import pandas as pd


def parameters_to_prune(model, model_type):
    pruned_parameters = []
    # todo: form a dict for different models
    if model_type == 'bert':
        layers = model.bert.encoder.layer
    elif model_type == 'roberta':
        layers = model.roberta.encoder.layer
    for layer in layers:
        params = [(layer.attention.self.key, 'weight'),
            (layer.attention.self.query, 'weight'),
            (layer.attention.self.value, 'weight'),
            (layer.attention.output.dense, 'weight'),
            (layer.intermediate.dense, 'weight'),
            (layer.output.dense, 'weight')]
        pruned_parameters.extend(params)

    pruned_parameters.append((model.classifier.dense,'weight'))
    pruned_parameters.append((model.classifier.out_proj,'weight'))
    return pruned_parameters
    # prune.global_unstructured(
    #     pruned_parameters,
    #     pruning_method=prune.L1Unstructured,
    #     amount=px,
    # )

def get_zero_rate(model, model_type):
    if model_type == 'bert':
        layers = model.bert.encoder.layer
    elif model_type == 'roberta':
        layers = model.roberta.encoder.layer
    sumlist, zerosum = 0, 0
    pruned_details = defaultdict(list)
    for ix, layer in enumerate(layers):
        query_all = float(layer.attention.self.query.weight_mask.nelement())
        query_zero = float(torch.sum(layer.attention.self.query.weight_mask == 0))
        pruned_details['query'].append(query_zero/query_all)

        key_all = float(layer.attention.self.key.weight_mask.nelement())
        key_zero = float(torch.sum(layer.attention.self.key.weight_mask == 0))
        pruned_details['key'].append(key_zero/key_all)

        value_all = float(layer.attention.self.value.weight_mask.nelement())
        value_zero = float(torch.sum(layer.attention.self.value.weight_mask == 0))
        pruned_details['value'].append(value_zero/value_all)
        
        attoutput_all = float(layer.attention.output.dense.weight_mask.nelement())
        attoutput_zero = float(torch.sum(layer.attention.output.dense.weight_mask == 0))
        pruned_details['att_output'].append(attoutput_zero/attoutput_all)
        
        
        int_dense_all = float(layer.intermediate.dense.weight_mask.nelement())
        int_dense_zero = float(torch.sum(layer.intermediate.dense.weight_mask == 0))
        pruned_details['int_dense'].append(int_dense_zero/int_dense_all)

        output_dense_all = float(layer.output.dense.weight_mask.nelement())
        output_dense_zero =float(torch.sum(layer.output.dense.weight_mask == 0))
        pruned_details['output_dense'].append(output_dense_zero/output_dense_all)

        num_zero = query_zero + key_zero + value_zero + attoutput_zero +int_dense_zero + output_dense_zero
        num_params = query_all + key_all + value_all + attoutput_all +int_dense_all + output_dense_all
        pruned_details['avg'].append(num_zero/num_params)
        zerosum += num_zero
        sumlist += num_params

        
        
    
    cls_dense_all = float(model.classifier.dense.weight_mask.nelement())
    sumlist += cls_dense_all
    cls_dense_zero = float(torch.sum(model.classifier.dense.weight_mask == 0))
    zerosum += cls_dense_zero
    pruned_details['cls_dense'].extend([cls_dense_zero/cls_dense_all]*len(pruned_details['avg']))

    out_proj_all = float(model.classifier.out_proj.weight_mask.nelement())
    sumlist += out_proj_all
    out_proj_zero = float(torch.sum(model.classifier.out_proj.weight_mask == 0))
    zerosum += out_proj_zero
    pruned_details['out_proj'].extend([out_proj_zero/out_proj_all]*len(pruned_details['avg']))
    pruned_details = pd.DataFrame.from_dict(pruned_details)
    return 100*zerosum / sumlist, pruned_details


def original_params(model):
    recover_dict = {}
    name_list = []
    ii = 0

    for ii in range(12):
        name_list.append('roberta.encoder.layer.'+str(ii)+'.attention.self.query.weight')
        name_list.append('roberta.encoder.layer.'+str(ii)+'.attention.self.key.weight')
        name_list.append('roberta.encoder.layer.'+str(ii)+'.attention.self.value.weight')
        name_list.append('roberta.encoder.layer.'+str(ii)+'.attention.output.dense.weight')
        name_list.append('roberta.encoder.layer.'+str(ii)+'.intermediate.dense.weight')
        name_list.append('roberta.encoder.layer.'+str(ii)+'.output.dense.weight')
    name_list.append('classifier.dense.weight')
    name_list.append('classifier.out_proj.weight')
        
    for key, value in model.named_parameters():

        # if 'roberta' in key:
        if key in name_list:
            new_key = key+'_orig'
        else:
            new_key = key

        recover_dict[new_key] = value

    return recover_dict