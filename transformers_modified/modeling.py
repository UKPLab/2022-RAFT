# coding=utf-8
# Copyright 2021 Intel Corporation. All rights reserved.
# DeepSpeed note, code taken from commit 3d59216cec89a363649b4fe3d15295ba936ced0f
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/modeling.py
# Deepspeed code taken from commit: 35b4582486fe096a5c669b6ca35a3d5c6a1ec56b
# https://github.com/microsoft/DeepSpeedExamples/tree/master/bing_bert
# RMS Norm taken from: https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/norms.py
#
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
from http.client import NOT_IMPLEMENTED
import logging
import math
import os
from re import I
import sys

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch.nn import CrossEntropyLoss, Module
from torch.nn.modules.loss import MSELoss
from torch.nn.parameter import Parameter
from torch.utils import checkpoint
from . import BertConfig, PreTrainedModel
from .modeling_outputs import SequenceClassifierOutput
from rational.torch import Rational 
from .modeling_outputs import QuestionAnsweringModelOutput

logger = logging.getLogger(__name__)


def get_deepspeed_config(args):
    if hasattr(args, "deepspeed") and args.deepspeed:
        from deepspeed import DeepSpeedConfig

        return DeepSpeedConfig(args.deepspeed)
    else:
        raise RuntimeError("deepspeed_config is not found in args.")


@torch.jit.script
def f_gelu(x):
    return F.gelu(x)


# @torch.jit.script
# def f_gelu(x):
#     pdtype = x.dtype
#     x = x.float()
#     y = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
#     return y.to(pdtype)


# @torch.jit.script
def bias_gelu(bias, y):
    x = bias + y
    # print(x.size())
    # print('we are', len(x.view(-1)))
    # print('smaller than -3',torch.sum(x<-3))
    res = F.gelu(x)
    # print('smaller than -3', torch.sum((res<0.01)&(res>-0.01))/len(res.view(-1)))
    return res 




# def bias_gelu(bias, y):
#     x = bias + y
#     return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


@torch.jit.script
def bias_relu(bias, y):
    x = bias + y
    return F.relu(x)


# @torch.jit.script
# def bias_gelu(bias, y):
#     x = bias + y
#     return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


@torch.jit.script
def bias_tanh(bias, y):
    x = bias + y
    return torch.tanh(x)


# ii=0
# import matplotlib.pyplot as plt
# import seaborn as sns
# fig, axs = plt.subplots(nrows=3, ncols=4)
# @torch.jit.script
def bias_rational(bias, y, rf):
    # global ii, fig, axs
    # print('after linear larger than 3',torch.sum((y>3))/len(y.view(-1).cpu().numpy()))
    x = bias + y
    # print('we are', len(x.view(-1)))
    # print('after linear smaller than -3',torch.sum(x<-3)/len(x.view(-1).cpu().numpy()))
    # print('larger than 3',torch.sum(x>3))
    res = rf(x)
    # sns.histplot(res.view(-1).cpu().numpy(), binwidth=0.01, binrange=[-1,1], ax=axs[ii//4, ii%4])
    # ii += 1
    # fig.savefig('./dist.png')
    # print(rf(torch.tensor(-4.0).cuda()))
    # print(rf(torch.tensor(100.0).cuda()))
    # exit()
    # print('smaller than -3', torch.sum((res<0.01)&(res>-0.01))/len(res.view(-1)))
    # print('we are', len(x.view(-1)))
    return res
    
    # return x
@torch.jit.script
def first_act(bias, x, weights):
    return F.linear(x, weights, bias)


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return f_gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": F.gelu, "relu": F.relu, "swish": swish, "tanh": F.tanh}


class LinearActivation(Module):
    r"""Fused Linear and activation Module."""
    __constants__ = ["bias"]

    def __init__(self, in_features, out_features, act="gelu", bias=True, use_deepnet=False):
        super(LinearActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fused_gelu = False
        self.fused_tanh = False
        self.fused_relu = False
        self.use_deepnet = use_deepnet
        # apex_ln = get_apex_layer_norm()
        # self.ln = apex_ln(out_features, eps=1e-12)
        # self.ln = nn.LayerNorm(in_features)
        if isinstance(act, str) or (sys.version_info[0] == 2 and isinstance(act, unicode)):
            if bias and act == "gelu":
                self.fused_gelu = True
            elif bias and act == "tanh":
                self.fused_tanh = True
            elif bias and act == "relu":
                self.fused_relu = True
            elif bias and act == "rational":
                self.rf = Rational(approx_func="gelu",version="A")
                self.fused_rational = True
            else:
                self.act_fn = ACT2FN[act]
        else:
            self.act_fn = act
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()


    def reset_parameters(self):
        if self.use_deepnet:
            self.weight = reset_parameters_scaled(self.weight)
        else:
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, inp):
        # print('before linear', torch.sum(inp<-3)/len(inp.view(-1).cpu().numpy()))

        if self.fused_gelu:
            return bias_gelu(self.bias, F.linear(inp, self.weight, None))
        elif self.fused_tanh:
            return bias_tanh(self.bias, F.linear(inp, self.weight, None))
        elif self.fused_relu:
            return bias_relu(self.bias, F.linear(inp, self.weight, None))
        elif self.fused_rational:
            res = bias_rational(self.bias, F.linear(inp, self.weight, None), self.rf)
            # res = first_act(self.bias, self.rf(inp), self.weight)
            return res
        else:
            return self.act_fn(F.linear(inp, self.weight, self.bias))

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class RegularLinearActivation(Module):
    """Regular Linear activation module with"""

    def __init__(self, in_features, out_features, act="gelu"):
        super(RegularLinearActivation, self).__init__()
        self.dense = nn.Linear(in_features, out_features)
        if isinstance(act, str) or (sys.version_info[0] == 2 and isinstance(act, unicode)):
            self.act = ACT2FN[act]

    def forward(self, hidden_states):
        return self.act(self.dense(hidden_states))


def get_apex_layer_norm():
    try:
        import apex

        # apex.amp.register_half_function(apex.normalization.fused_layer_norm, 'FusedLayerNorm')
        import apex.normalization

        # apex.amp.register_float_function(apex.normalization.FusedLayerNorm, 'forward')
        return apex.normalization.FusedLayerNorm
    except ImportError:
        pass
        # raise Exception(f"Layer norm of type apex is not available, apex not installed.")


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, p=-1.0, eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param dim: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = dim
        self.p = p
        self.bias = bias

        self.scale = torch.nn.Parameter(torch.ones(dim))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = torch.nn.Parameter(torch.zeros(dim))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


LAYER_NORM_TYPES = {"pytorch": nn.LayerNorm, "apex": get_apex_layer_norm(), "rms_norm": RMSNorm}


def get_layer_norm_type(config):
    if config.layer_norm_type in LAYER_NORM_TYPES:
        return LAYER_NORM_TYPES[config.layer_norm_type]
    else:
        raise Exception(f"Layer norm of type {config.layer_norm_type} is not available.")


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.layernorm_embedding = config.layernorm_embedding
        if config.layernorm_embedding:
            BertLayerNorm = get_layer_norm_type(config)
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        if self.layernorm_embedding:
            embeddings = self.LayerNorm(embeddings)

        embeddings = self.dropout(embeddings)
        return embeddings

def reset_parameters_scaled(weights, n=12):
    return init.xavier_normal_(weights, gain=(8*n)**(-0.25))

def reset_xavier_parameters(weights):
    return init.xavier_normal_(weights, gain=1)

def deepnorm(x, fx, n=12):
    return x*(2*n)**0.25 + fx

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        if config.use_deepnet:
            self.value.weight = reset_parameters_scaled(self.value.weight)
            self.key.weight = reset_xavier_parameters(self.key.weight)
            self.query.weight = reset_xavier_parameters(self.query.weight)
    


        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)
        

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_key_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 3, 1)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense.bert_output_layer = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.use_deepnet:
            self.dense.weight = reset_parameters_scaled(self.dense.weight)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        context_layer, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(context_layer, input_tensor)
        output = (
            attention_output,
            attention_probs,
        )
        return output



class BertIntermediate(nn.Module):
    def __init__(self, config, use_rational=False):
        super(BertIntermediate, self).__init__()
        if config.fused_linear_layer:
            linear_layer = LinearActivation
        else:
            linear_layer = RegularLinearActivation
        if use_rational:
            self.dense_act = linear_layer(config.hidden_size, config.intermediate_size, act="rational", use_deepnet=config.use_deepnet)
        else:
            self.dense_act = linear_layer(
                config.hidden_size, config.intermediate_size, act=config.hidden_act
            )

    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dense.bert_output_layer = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.use_deepnet:
            self.dense.weight = reset_parameters_scaled(self.dense.weight)
        # self.ln = nn.LayerNorm(config.intermediate_size)
        # if use_rational:
        #     self.act = Rational()

    def forward(self, hidden_states):
        # hidden_states = self.ln(hidden_states)
        # hidden_states = self.act(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states



class BertLayer(nn.Module):
    def __init__(self, config, use_rational=False):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.config = config

        BertLayerNorm = get_layer_norm_type(config)

        self.PreAttentionLayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.PostAttentionLayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.intermediate = BertIntermediate(config, use_rational)
        self.output = BertOutput(config)

    def maybe_layer_norm(self, hidden_states, layer_norm, current_ln_mode):
        if self.config.useLN and self.config.encoder_ln_mode in current_ln_mode:
            return layer_norm(hidden_states)
        else:
            return hidden_states

    def forward(self, hidden_states, attention_mask, action=1, keep_prob=1.0, zero_out=False):
        attention_probs = None
        intermediate_input = None

        if action == 0:
            intermediate_input = hidden_states
        else:
            pre_attn_input = self.maybe_layer_norm(
                hidden_states, self.PreAttentionLayerNorm, "pre-ln"
            )
            self_attn_out = self.attention(pre_attn_input, attention_mask)

            attention_output, attention_probs = self_attn_out
            attention_output = attention_output * 1 / keep_prob

            if self.config.use_deepnet:
                intermediate_input = deepnorm(hidden_states, attention_output)
            else:
                intermediate_input = hidden_states + attention_output
            
            intermediate_input = self.maybe_layer_norm(
                intermediate_input, self.PreAttentionLayerNorm, "post-ln"
            )

        if action == 0:
            layer_output = intermediate_input
        else:
            intermediate_pre_ffn = self.maybe_layer_norm(
                intermediate_input, self.PostAttentionLayerNorm, "pre-ln"
            )
            if not zero_out:
                # print('no zero out...')
                intermediate_output = self.intermediate(intermediate_pre_ffn)
                layer_output = self.output(intermediate_output)
                layer_output = layer_output * 1 / keep_prob

            # inter_output = abs(layer_output)
                if self.config.use_deepnet:
                    layer_output = deepnorm(intermediate_input, layer_output)
                else:
                    layer_output = layer_output + intermediate_input
            else:
                print('zero out...')
                layer_output = intermediate_input*(2*12)**0.25
            
            layer_output = self.maybe_layer_norm(
                layer_output, self.PostAttentionLayerNorm, "post-ln"
            )
            # assert inter_output!= layer_output
        output = (
            layer_output,
            attention_probs,
            # torch.mean(inter_output)
        )
        return output


class BertEncoder(nn.Module):
    def __init__(self, config, args):
        super(BertEncoder, self).__init__()
        self.config = config
        BertLayerNorm = get_layer_norm_type(config)
        self.FinalLayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.is_transformer_kernel = (
            hasattr(args, "deepspeed_transformer_kernel") and args.deepspeed_transformer_kernel
        )

        if hasattr(args, "deepspeed_transformer_kernel") and args.deepspeed_transformer_kernel:
            from deepspeed import DeepSpeedTransformerConfig, DeepSpeedTransformerLayer

            # ds_config = get_deepspeed_config(args)
            # has_huggingface = hasattr(args, "huggingface")
            ds_transformer_config = DeepSpeedTransformerConfig(
                batch_size=args.hf_deepspeed_config.config.get("train_micro_batch_size_per_gpu"),
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                heads=config.num_attention_heads,
                attn_dropout_ratio=config.attention_probs_dropout_prob,
                hidden_dropout_ratio=config.hidden_dropout_prob,
                num_hidden_layers=config.num_hidden_layers,
                initializer_range=config.initializer_range,
                local_rank=args.local_rank if hasattr(args, "local_rank") else -1,
                seed=args.seed,
                fp16=True,
                pre_layer_norm=True if "pre-ln" in config.encoder_ln_mode else False,
                normalize_invertible=args.normalize_invertible,
                gelu_checkpoint=args.gelu_checkpoint,
                adjust_init_range=True,
                attn_dropout_checkpoint=args.attention_dropout_checkpoint,
                stochastic_mode=args.stochastic_mode,
                training=self.training,
            )

            self.layer = nn.ModuleList(
                [
                    copy.deepcopy(DeepSpeedTransformerLayer(ds_transformer_config))
                    for _ in range(config.num_hidden_layers)
                ]
            )
        else:
            # layer = BertLayer(config)
            self.layer = nn.ModuleList(
                [BertLayer(config, use_rational= str(_) in config.rational_layers) for _ in range(self.config.num_hidden_layers)]
            )

    def add_attention(self, all_attentions, attention_probs):
        if attention_probs is not None:
            all_attentions.append(attention_probs)

        return all_attentions

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_all_encoded_layers=False,
        checkpoint_activations=False,
        output_attentions=False,
    ):
        all_encoder_layers = []
        all_attentions = []
        all_inter_output = None

        def custom(start, end):
            def custom_forward(*inputs):
                layers = self.layer[start:end]
                x_ = inputs[0]
                for layer in layers:
                    x_ = layer(x_, inputs[1])
                return x_

            return custom_forward

        if checkpoint_activations:
            l = 0
            num_layers = len(self.layer)
            chunk_length = math.ceil(math.sqrt(num_layers))
            while l < num_layers:
                hidden_states = checkpoint.checkpoint(
                    custom(l, l + chunk_length), hidden_states, attention_mask * 1
                )
                l += chunk_length
            # decoder layers
        else:
            for ix, layer_module in enumerate(self.layer):
                if self.is_transformer_kernel:
                    # using Deepspeed Transformer kernel
                    hidden_states = layer_module(hidden_states, attention_mask)
                else:
                    layer_out = layer_module(
                        hidden_states,
                        attention_mask,
                        # zero_out= (ix==8) or (ix==9) or (ix==10)
                    )
                    # hidden_states, attention_probs, inter_output = layer_out
                    hidden_states, attention_probs = layer_out

                    # get all attention_probs from layers
                    if output_attentions:
                        all_attentions = self.add_attention(all_attentions, attention_probs)

                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
                # inter_output = inter_output.view(-1,)
                # all_inter_output = inter_output if all_inter_output is None else torch.cat([all_inter_output, inter_output])
        if not output_all_encoded_layers or checkpoint_activations:
            if self.config.useLN and self.config.encoder_ln_mode in "pre-ln":
                hidden_states = self.FinalLayerNorm(hidden_states)

            all_encoder_layers.append(hidden_states)
        outputs = (all_encoder_layers,)
        if output_attentions:
            outputs += (all_attentions,)
        # print('aa', all_inter_output)
        # outputs += (all_inter_output,)
        return outputs


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        if config.fused_linear_layer:
            linear_layer = LinearActivation
        else:
            linear_layer = RegularLinearActivation

        if 'pooler' in config.rational_layers:
            act="rational"
        else:
            act="tanh"
        self.dense_act = linear_layer(config.hidden_size, config.hidden_size, act=act)

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense_act(first_token_tensor)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.config = config
        if config.fused_linear_layer:
            linear_layer = LinearActivation
        else:
            linear_layer = RegularLinearActivation
        self.dense_act = linear_layer(config.hidden_size, config.hidden_size, act=config.hidden_act)
        BertLayerNorm = get_layer_norm_type(config)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
        if self.config.useLN:
            hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1), bert_model_embedding_weights.size(0), bias=False
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))
        # self.sparse_predict = config.sparse_mask_prediction
        # if not config.sparse_mask_prediction:
        #     self.decoder.bias = self.bias

    def forward(self, hidden_states, masked_token_indexes, sparse_predict=True):
        if sparse_predict:
            if masked_token_indexes is not None:
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])[
                    masked_token_indexes
                ]
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        # if not self.sparse_predict:
        #     hidden_states = torch.index_select(
        #         hidden_states.view(-1, hidden_states.shape[-1]), 0, masked_token_indexes
        #     )
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output, masked_token_indexes=None, sparse_predict=True):
        prediction_scores = self.predictions(sequence_output, masked_token_indexes, sparse_predict)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output, masked_token_indexes=None):
        prediction_scores = self.predictions(sequence_output, masked_token_indexes)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    base_model_prefix = "bert"
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, inputs=inputs, kwargs=kwargs)
        # print(inputs)
        # print(kwargs)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, args=None, add_pooling_layer=True):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        # set pad_token_id that is used for sparse attention padding
        self.pad_token_id = (
            config.pad_token_id
            if hasattr(config, "pad_token_id") and config.pad_token_id is not None
            else 0
        )
        self.encoder = BertEncoder(config, args)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        logger.info("Init BERT pretrain model")

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=False,
        checkpoint_activations=False,
        output_attentions=False,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        extended_attention_mask = extended_attention_mask.to(
            dtype=self.embeddings.word_embeddings.weight.dtype  # should be of same dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)

        encoder_output = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            checkpoint_activations=checkpoint_activations,
            output_attentions=output_attentions,
        )
        encoded_layers = encoder_output[0]
        sequence_output = encoded_layers[-1]

        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        output = (
            encoded_layers,
            pooled_output,
        )
        if output_attentions:
            output += (encoder_output[1],)
        # # output of intermeidate
        # if pooled_output is not None:
        #     intermediate = torch.cat([encoder_output[-1], torch.mean(abs(pooled_output)).view(-1,)])
        # else:
        #     intermediate = None
        # output += (intermediate,)
        return output


class BertForPreTraining(BertPreTrainedModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, args):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config, args)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, batch):
        pass
        # input_ids = batch[1]
        # token_type_ids = batch[3]
        # attention_mask = batch[2]
        # masked_lm_labels = batch[5]
        # next_sentence_label = batch[4]
        # checkpoint_activations = False

        # sequence_output, pooled_output = self.bert(
        #     input_ids,
        #     token_type_ids,
        #     attention_mask,
        #     output_all_encoded_layers=False,
        #     checkpoint_activations=checkpoint_activations,
        # )

        # if masked_lm_labels is not None and next_sentence_label is not None:
        #     # filter out all masked labels.
        #     masked_token_indexes = torch.nonzero(
        #         (masked_lm_labels + 1).view(-1), as_tuple=False
        #     ).view(-1)
        #     prediction_scores, seq_relationship_score = self.cls(
        #         sequence_output, pooled_output, masked_token_indexes
        #     )
        #     target = torch.index_select(masked_lm_labels.view(-1), 0, masked_token_indexes)

        #     loss_fct = CrossEntropyLoss(ignore_index=-1)
        #     masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), target)
        #     next_sentence_loss = loss_fct(
        #         seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
        #     )
        #     total_loss = masked_lm_loss + next_sentence_loss
        #     return total_loss
        # else:
        #     prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        #     return prediction_scores, seq_relationship_score


class BertLMHeadModel(BertPreTrainedModel):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is  not `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertMultiHead(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    """
    
    def __init__(self, config, args):
        super(BertLMHeadModel, self).__init__(config)
        self.bert = BertModel(config, args)

        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.sparse_predict = config.sparse_mask_prediction
        self.init_weights()
    
    def set_sparse_predict(self):
        if self.training:
            self.sparse_predict = True
        else:
            self.sparse_predict = False

    def forward(self, input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_hidden_states=None,
        return_dict=None,
        output_attentions=False):


        checkpoint_activations = False

        bert_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )
        sequence_output = bert_output[0]

        if labels is None:
            prediction_scores = self.cls(sequence_output)
            return prediction_scores

        masked_token_indexes = torch.nonzero((labels + 100).view(-1), as_tuple=False).view(
            -1
        )

        prediction_scores = self.cls(sequence_output, masked_token_indexes=masked_token_indexes, sparse_predict=self.sparse_predict)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if self.sparse_predict:
                labels = torch.index_select(labels.view(-1), 0, masked_token_indexes)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (masked_lm_loss,prediction_scores)
            if output_attentions:
                outputs += (bert_output[-1],)
            return outputs
        else:
            return prediction_scores


class BertForNextSentencePrediction(BertPreTrainedModel):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, args):
        super(BertForNextSentencePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        next_sentence_label=None,
        checkpoint_activations=False,
    ):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False
        )
        seq_relationship_score = self.cls(pooled_output)

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )
            return next_sentence_loss
        else:
            return seq_relationship_score


class BertForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, args=None):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,
    ):

        outputs = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs[0],
            attentions=None,
        )


class BertForQuestionAnswering(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_all_encoded_layers = False
        )
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

