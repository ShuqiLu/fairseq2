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
# from layer_norm import LayerNorm
# from multihead_attention import MultiheadAttention
# from positional_embedding import PositionalEmbedding
# from transformer_sentence_encoder_layer import TransformerSentenceEncoderLayer
import random

#import model_utils as utils
#from fairseq import utils
# from fairseq.models import (
#     FairseqDecoder,
#     FairseqLanguageModel,
#     #register_model,
#     #register_model_architecture,
# )
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.roberta import RobertaModel

#from .hub_interface import RobertaHubInterface

# import dgl
# import dgl.function as fn
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
    def __init__(
        self,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 24,
        embedding_dim: int = 1024,
        ffn_embedding_dim: int = 4096,
        num_attention_heads: int = 16,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        layerdrop : float = 0.0,
        max_seq_len: int = 512, 
        num_segments: int = 0,
        use_position_embeddings: bool = True,
        offset_positions_by_padding: bool = True,
        encoder_normalize_before: bool = True,
        apply_bert_init: bool = True,
        activation_fn: str = "gelu",
        learned_pos_embedding: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        embed_scale: float = None,
        freeze_embeddings: bool = False, 
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
    ):

        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.apply_bert_init = apply_bert_init
        self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable
        self.num_encoder_layers=num_encoder_layers
        self.num_attention_heads=num_attention_heads

        self.embed_tokens = nn.Embedding(
            self.vocab_size, self.embedding_dim, self.padding_idx 
        )



        self.embed_scale = embed_scale

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        self.segment_embeddings = (
            nn.Embedding(self.num_segments, self.embedding_dim, padding_idx=None)
            if self.num_segments > 0
            else None
        )


        self.embed_positions = ( 
            PositionalEmbedding(
                self.max_seq_len,
                self.embedding_dim,
                padding_idx=(self.padding_idx if offset_positions_by_padding else None),
                #padding_idx=None,
                learned=self.learned_pos_embedding,
            )
            if self.use_position_embeddings
            else None
        )

        self.layers = nn.ModuleList([
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    # add_bias_kv=add_bias_kv,
                    # add_zero_attn=add_zero_attn,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                    export=export,
                )
                for _ in range(self.num_encoder_layers)
            ]
            )
        #self.roberta = torch.hub.load('pytorch/fairseq', load_model)
        # self.roberta = RobertaModel.from_pretrained('model/roberta.base/',checkpoint_file='model.pt')
        # self.roberta=RobertaModel()
        # print(self.roberta.encode('Hello world!'))

        #self.score = nn.Linear(embedding_dim*2, 1, bias=True)

        self.score2 = nn.Sequential(nn.Linear(embedding_dim*2, 200, bias=True),nn.Tanh()) 

        self.score3 = nn.Linear(200, 1, bias=True)

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        if self.apply_bert_init:
            self.apply(init_bert_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            freeze_module_params(self.embed_tokens)
            freeze_module_params(self.segment_embeddings)
            freeze_module_params(self.embed_positions)
            freeze_module_params(self.emb_layer_norm)

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])


   
    def forward(self, his_id , candidate_id , label,  rank_mask=None, gpu_tracker=None,last_state_only: bool = True):#

        
        # batch_size,can_num,can_legth=candidate_id.shape
        # batch_size,_,his_length=his_id.shape

        batch_size,can_num,can_legth=candidate_id.shape
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
        # his_features=his_features.reshape(batch_size,1,his_features.shape[-1])
        # his_features=his_features.transpose(1,2).repeat(1,1,can_num).transpose(1,2)

        can_features=self.extract_features(l,can_padding_mask)
        can_features=can_features[:,0,:]

        can_features=can_features.reshape(batch_size,can_num,can_features.shape[-1])


        # features=torch.cat( (his_features,can_features ) ,2)
        # # print('features: ',features)

        # # print('dense: ',self.score2(features))

        # res1=self.score2(features)
        # res=self.score3(res1)
        res=torch.matmul(his_features,can_features.transpose(1,2))

        #res=res.squeeze(-1)
        res=res.reshape(-1,2)

        #print('???',res.shape,label.shape)
        


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


    def extract_features(self,h,padding_mask,last_state_only=True):

        h=h.transpose(0,1)
        for layer in self.layers:                
            # dropout_probability = random.uniform(0, 1)
            # if not self.training or (dropout_probability > self.layerdrop):
            #print('???',h.shape)
            h, _ = layer(h, self_attn_padding_mask=padding_mask, self_attn_mask=None)
            
        return h.transpose(0,1)
        

    def predict(self,his_id , candidate_id):

        batch_size,can_num,can_legth=candidate_id.shape
        his_batch,his_num,his_length=his_id.shape
        sample_size=candidate_id.shape[0]
        #print('in model: ',his_id.shape,candidate_id.shape,rank_mask.shape,label.shape,label)
        #print(his_id)
        his_id=his_id.reshape(-1,his_id.shape[-1])
        candidate_id=candidate_id.reshape(-1,can_legth)
        #print('candidate_id',candidate_id)
        # print('candidate_id: ',candidate_id)
        # print('candidate_id2: ',candidate_id2)

        his_padding_mask = his_id.eq(self.padding_idx)#
        can_padding_mask=candidate_id.eq(self.padding_idx)

        #can_padding_mask2=candidate_id2.eq(self.padding_idx)
        #print('can_padding_mask',can_padding_mask)

        #print('can_padding_mask: ',can_padding_mask)
        
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

            #l2 = self.emb_layer_norm(l2)

        h = F.dropout(h, p=self.dropout, training=self.training)#

        l = F.dropout(l, p=self.dropout, training=self.training)

        #l2 = F.dropout(l2, p=self.dropout, training=self.training)

        if his_padding_mask is not None:
            h *= 1 - his_padding_mask.unsqueeze(-1).type_as(h)#

        if can_padding_mask is not None:
            l *= 1 - can_padding_mask.unsqueeze(-1).type_as(l)


        his_features = self.extract_features(h,his_padding_mask)#bsz,length,dim
        his_features=his_features[:,0,:]

        his_features=his_features.reshape(1,his_features.shape[-1])
        his_features=his_features.transpose(0,1).repeat(1,can_num*batch_size).transpose(0,1)

        #print('???his_features: ',his_features)        

        can_features=self.extract_features(l,can_padding_mask)
        can_features=can_features[:,0,:]

        can_features=can_features.reshape(-1,can_features.shape[-1])

        features=torch.cat( (his_features,can_features ) ,1)
        # print('features: ',features)

        # print('dense: ',self.score2(features))

        res1=self.score2(features)
        res=self.score3(res1)

        res=res.squeeze(-1)
        #print('res: ',res)

        res=F.sigmoid(res)
        #print('res: ',res)
        #print('res shape: ',res.shape)

        return res



















        










        




