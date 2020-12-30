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

random.seed(1)
np.random.seed(1) 
torch.manual_seed(1) 
torch.cuda.manual_seed(1)


class Plain_bert(nn.Module):#
    def __init__(self,args):
        super().__init__()
        embedding_dim=768
        self.roberta=RobertaModel.from_pretrained(os.path.join(args.data_dir,'roberta.base'), checkpoint_file='model.pt')
        self.dense = nn.Sequential(nn.Linear(embedding_dim, embedding_dim, bias=True),nn.Tanh()) 
        self.layer_norm = LayerNorm(embedding_dim)

        # self.layers = nn.ModuleList([
        #         TransformerSentenceEncoderLayer(
        #             embedding_dim=self.embedding_dim,
        #             ffn_embedding_dim=ffn_embedding_dim,
        #             num_attention_heads=num_attention_heads,
        #             dropout=self.dropout,
        #             attention_dropout=attention_dropout,
        #             activation_dropout=activation_dropout,
        #             activation_fn=activation_fn,
        #             # add_bias_kv=add_bias_kv,
        #             # add_zero_attn=add_zero_attn,
        #             q_noise=q_noise,
        #             qn_block_size=qn_block_size,
        #             export=export,
        #         )
        #         for _ in range(self.num_encoder_layers)
        #     ]
        #     )
        
    def forward(self, his_id , candidate_id , label):#
        # batch_size,can_num,can_legth=candidate_id.shape
        # batch_size,_,his_length=his_id.shape
        print('???')
        batch_size,can_num,can_legth=candidate_id.shape
        batch_size,_,his_length=his_id.shape
        sample_size=candidate_id.shape[0]
        his_id=his_id.reshape(-1,his_id.shape[-1])
        candidate_id=candidate_id.reshape(-1,can_legth)


        his_features = self.roberta.decoder(his_id)#bsz,length,dim
        his_features=his_features[:,0,:]

        his_features=his_features.reshape(batch_size,1,his_features.shape[-1])
        #his_features=his_features.transpose(1,2).repeat(1,1,can_num).transpose(1,2)
        can_features=self.roberta(candidate_id)
        can_features=can_features[:,0,:]

        can_features=can_features.reshape(batch_size,can_num,can_features.shape[-1])       
        res=torch.matmul(his_features,can_features.transpose(1,2))

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

        return loss,torch.tensor(sample_size).cuda()
        

    def predict(self,his_id , candidate_id):

        batch_size,can_num,can_legth=candidate_id.shape
        #print('???',candidate_id.shape)
        batch_size,_,his_length=his_id.shape
        sample_size=candidate_id.shape[0]
        #print('in model: ',his_id.shape,candidate_id.shape,rank_mask.shape,label.shape,label)
        #print(his_id)
        his_id=his_id.reshape(-1,his_id.shape[-1])
        candidate_id=candidate_id.reshape(-1,can_legth)

        his_padding_mask = his_id.eq(self.padding_idx)#
        can_padding_mask=candidate_id.eq(self.padding_idx)

       
        if not self.traceable and not his_padding_mask.any():
            his_padding_mask = None

        if not self.traceable and not can_padding_mask.any():
            can_padding_mask = None

        # if not self.traceable and not can_padding_mask2.any():
        #     can_padding_mask2 = None

        h = self.embed_tokens(his_id)
        l = self.embed_tokens(candidate_id)


        if self.embed_scale is not None:
            h *= self.embed_scale
            l *= self.embed_scale

            #l2 *= self.embed_scale

        if self.embed_positions is not None:
            h += self.embed_positions(his_id)
            l += self.embed_positions(candidate_id)

        if self.quant_noise is not None:
            print('error!')
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            h = self.emb_layer_norm(h)
            l = self.emb_layer_norm(l)

        h = F.dropout(h, p=self.dropout, training=self.training)#

        l = F.dropout(l, p=self.dropout, training=self.training)

        #l2 = F.dropout(l2, p=self.dropout, training=self.training)

        if his_padding_mask is not None:
            h *= 1 - his_padding_mask.unsqueeze(-1).type_as(h)#

        if can_padding_mask is not None:
            l *= 1 - can_padding_mask.unsqueeze(-1).type_as(l)

        his_features = self.extract_features(h,his_padding_mask)#bsz,length,dim
        his_features=his_features[:,0,:]

        his_features=his_features.reshape(batch_size,1,his_features.shape[-1])
        #his_features=his_features.transpose(1,2).repeat(1,1,can_num).transpose(1,2)

        can_features=self.extract_features(l,can_padding_mask)
        can_features=can_features[:,0,:]

        can_features=can_features.reshape(batch_size,can_num,can_features.shape[-1])


        his_features = self.dense(his_features)
        his_features = self.layer_norm(his_features)

        can_features = self.dense(can_features)
        can_features = self.layer_norm(can_features)



        #features=torch.cat( (his_features,can_features ) ,2)
        # print('features: ',features)

        # print('dense: ',self.score2(features))

        # res1=self.score2(features)
        # res=self.score3(res1)

        # res=res.squeeze(-1)
        res=torch.matmul(his_features,can_features.transpose(1,2))


        res=res.reshape(-1)
        #print('res: ',res)

        #res=F.sigmoid(res)
        #print('res: ',res)
        #print('res shape: ',res.shape)

        return res



















        










        




