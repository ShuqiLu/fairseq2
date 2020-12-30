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
import random
import os

from fairseq.modules.transformer_sentence_encoder import init_bert_params
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


class Plain_bert(nn.Module):#
    def __init__(self,args):
        super().__init__()
        embedding_dim=768

        self.dense = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = LayerNorm(embedding_dim)
        init_bert_params(self.dense)
        self.encoder=TransformerSentenceEncoder(
                padding_idx=1,
                vocab_size=32769,
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
        
    def forward(self, his_id , candidate_id , label,mode='train'):#
        # batch_size,can_num,can_legth=candidate_id.shape
        # batch_size,_,his_length=his_id.shape
        batch_size,can_num,can_legth=candidate_id.shape
        batch_size,_,his_length=his_id.shape
        sample_size=candidate_id.shape[0]
        his_id=his_id.reshape(-1,his_id.shape[-1])
        candidate_id=candidate_id.reshape(-1,can_legth)


        his_features,_ = self.encoder(his_id)#bsz,length,dim
        his_features=his_features[-1].transpose(0,1)[:,0,:]
        his_features=his_features.reshape(batch_size,1,his_features.shape[-1])
        #his_features=his_features.transpose(1,2).repeat(1,1,can_num).transpose(1,2)
        can_features,_=self.encoder(candidate_id)
        can_features=can_features[-1].transpose(0,1)[:,0,:]
        can_features=can_features.reshape(batch_size,can_num,can_features.shape[-1]) 


        his_features = self.dense(his_features)
        his_features = self.layer_norm(his_features)

        can_features = self.dense(can_features)
        can_features = self.layer_norm(can_features)


        res=torch.matmul(his_features,can_features.transpose(1,2))
        if mode !='train':
            return res.reshape(-1)#,label.view(-1)

        res=res.reshape(-1,2)
        #print('???',res,sample_size)

        loss = F.nll_loss(
            F.log_softmax(
                res.view(-1, res.size(-1)),
                dim=-1,
                dtype=torch.float32,
            ),
            label.view(-1),
            reduction='sum',
            #ignore_index=self.padding_idx,
        )
        #print('loss: ',loss)
        return loss#,torch.tensor(sample_size).cuda()
        

    def predict(self,his_id , candidate_id):

        batch_size,can_num,can_legth=candidate_id.shape
        batch_size,_,his_length=his_id.shape
        sample_size=candidate_id.shape[0]
        his_id=his_id.reshape(-1,his_id.shape[-1])
        candidate_id=candidate_id.reshape(-1,can_legth)


        his_features,_ = self.encoder(his_id)#bsz,length,dim
        his_features=his_features[-1].transpose(0,1)[:,0,:]
        his_features=his_features.reshape(batch_size,1,his_features.shape[-1])
        #his_features=his_features.transpose(1,2).repeat(1,1,can_num).transpose(1,2)
        can_features,_=self.encoder(candidate_id)
        can_features=can_features[-1].transpose(0,1)[:,0,:]
        can_features=can_features.reshape(batch_size,can_num,can_features.shape[-1]) 


        his_features = self.dense(his_features)
        his_features = self.layer_norm(his_features)

        can_features = self.dense(can_features)
        can_features = self.layer_norm(can_features)


        res=torch.matmul(his_features,can_features.transpose(1,2))


        #res=res.reshape(-1)
        res=res.squeeze(1)
        #print('res: ',res)

        #res=F.sigmoid(res)
        #print('res: ',res)
        #print('res shape: ',res.shape)

        return res



















        










        




