# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
    TransformerSentenceEncoderLayer,
)

from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq import utils

import random
import os

from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.models.roberta import RobertaModel
import torch
import torch.optim as optim
#from torch.utils.data import DataLoader
import numpy as np
#from fairseq.models.roberta import RobertaModel
from fairseq.modules import (
    LayerNorm,
    TransformerSentenceEncoder,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from fairseq.models.transformer import TransformerDecoder
from fairseq.models.fairseq_encoder import EncoderOut
random.seed(1)
np.random.seed(1) 
torch.manual_seed(1) 
torch.cuda.manual_seed(1)


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.v_proj.weight.data.normal_(mean=0.0, std=0.02)


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target)

    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        #print('target: ',target.shape,' pad_mask: ',pad_mask.shape,' smooth_loss: ',smooth_loss.shape,' nll_loss: ',nll_loss.shape)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout, q_noise=0, qn_block_size=8):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None ):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)


        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))
        

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the unmasked tokens while training,
        # saves both memory and computation
        #print('???features: ',features.shape,masked_tokens)

        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        #print('???forward: ',x.shape,self.weight.shape)
        x = F.linear(x, self.weight) + self.bias
        return x

class Plain_bert(nn.Module):#
    def __init__(self,args,dictionary):
        super().__init__()
        embedding_dim=768
        self.padding_idx=1

        self.dense = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = LayerNorm(embedding_dim)
        init_bert_params(self.dense)
        self.encoder=TransformerSentenceEncoder(
                padding_idx=1,
                vocab_size=50265,
                num_encoder_layers=12,
                embedding_dim=768,
                ffn_embedding_dim=3072,
                num_attention_heads=12,
                dropout=0.1,
                attention_dropout=0.1,
                activation_dropout=0.0,
                layerdrop=0.0,
                max_seq_len=512,
                num_segments=0,
                encoder_normalize_before=True,
                apply_bert_init=True,
                activation_fn="gelu",
                q_noise=0.0,
                qn_block_size=8,
        )
        embed_tokens=self.encoder.embed_tokens
        self.lm_head = RobertaLMHead(
            embed_dim=embedding_dim,
            output_dim=50265,
            activation_fn="gelu",
            weight=embed_tokens.weight,
        )

        #args=base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = 512
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = 512
        print('???',embed_tokens.embedding_dim)

        self.decoder=TransformerDecoder(args, dictionary, embed_tokens, no_encoder_attn=getattr(args, "no_cross_attention", False))

        self.class_num=args.num_classes
        self.classification_heads = RobertaClassificationHead(
            768,
            768,
            self.class_num,
            'tanh',
            0.0,
            0.0,
            8,
        )

        
    def forward(self, token_id, mask_label=None, decode_label=None, label=None):#
        # batch_size,can_num,can_legth=candidate_id.shape
        # batch_size,_,his_length=his_id.shape

        #print('???shape: ',token_id.shape,mask_label.shape,decode_label.shape)
        if label is not None:
            return self.predict(token_id,label)


        token_features,_ = self.encoder(token_id)#bsz,length,dim
        token_features=token_features[-1].transpose(0,1)#[:,0,:]
        loss_mask, sample_size_mask = self.predict_mask(token_features, mask_label)


        h=token_features[:,0:,]
        h=EncoderOut(
            encoder_out=h,  # T x B x C
            encoder_padding_mask=None,  # B x T
            encoder_embedding=None,  # B x T x C
            encoder_states=None,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

        loss_decode, sample_size_decode =self.predict_decode(h ,decode_label) 



        # loss = F.nll_loss(
        #     F.log_softmax(
        #         res.view(-1, res.size(-1)),
        #         dim=-1,
        #         dtype=torch.float32,
        #     ),
        #     label.view(-1),
        #     reduction='sum',
        #     #ignore_index=self.padding_idx,
        # )

        #loss=0.5*loss_decode+0.5*loss_mask

        # loss=loss_mask
        # sample_size= sample_size_mask

        # loss=loss_decode
        # sample_size= sample_size_decode

        #return loss, sample_size #,torch.tensor(sample_size).cuda()
        return loss_mask,sample_size_mask,loss_decode,sample_size_decode



    def predict_mask(self,h,mask_label):

        masked_tokens = mask_label.ne(self.padding_idx)

        sample_size = masked_tokens.int().sum().item()
        if sample_size == 0:
            masked_tokens=None
            return torch.tensor(0.).cuda(),0
        logits =  self.lm_head(h, masked_tokens)

        targets=mask_label
        if sample_size != 0:
            targets=targets[masked_tokens]

        loss = F.nll_loss(
            F.log_softmax(
                logits.view(-1, logits.size(-1)),
                dim=-1,
                dtype=torch.float32,
            ),
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )

        # if sample_size!=0:
        #     loss_t=loss/sample_size
        # else:
        #     loss_t=0
        
        return loss,sample_size

    def predict_decode(self,h,decode_label):

        masked_tokens = decode_label.ne(self.padding_idx)
        sample_size = masked_tokens.int().sum().item()

        if sample_size == 0:
            masked_tokens=None
            return torch.tensor(0.).cuda(),0

        src_token=decode_label[:,:-1]
        tgt_token=decode_label[:,1:]
        #print(src_token.shape,h.shape)
        tgt_token=tgt_token.reshape(-1)
        decoder_output=self.decoder(src_token,encoder_out=h)
        #print('???',decoder_output[0].shape)
        lprobs= F.log_softmax(decoder_output[0], dim=-1, dtype=torch.float32)# utils.log_softmax(decoder_output, dim=-1, onnx_trace=False)
        #print('???',lprobs.shape)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        #print('???',lprobs.shape)

        target=tgt_token

        eps=0.

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, eps, ignore_index=self.padding_idx, reduce=True,
        )

        #loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        return loss, sample_size



    def predict(self,token_id,label):
        token_features,_ = self.encoder(token_id)#bsz,length,dim
        token_features=token_features[-1].transpose(0,1)#[:,0,:]
        sample_size = len(label)
        h=token_features
        #h=h[:,0,:]
        logits =  self.classification_heads(h)
        targets=label.view(-1)
        if self.class_num==1:
            logits = logits.view(-1).float()
            targets = targets.float()
            loss = F.mse_loss(
                logits,
                targets,
                reduction='sum',
            )
            return loss,sample_size,0
        else:
            preds = logits.argmax(dim=1)
            acc=(preds == targets).sum()
            loss = F.nll_loss(
                F.log_softmax(
                    logits.view(-1, logits.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                ),
                targets.view(-1),
                reduction='sum',
                #ignore_index=self.padding_idx,
            )
        
            # logits=F.softmax(
            #         logits.view(-1, logits.size(-1)),
            #         dim=-1,
            #         dtype=torch.float32,
            #     )
            return loss,sample_size,acc



        



















        










        




