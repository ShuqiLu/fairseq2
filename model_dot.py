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
        embedding_dim=768
    ):

        super().__init__()
        self.dense = nn.Sequential(nn.Linear(embedding_dim, embedding_dim, bias=True),nn.Tanh()) 
        self.layer_norm = LayerNorm(embedding_dim)

    def forward(self, his_features , can_features , label,  rank_mask=None, gpu_tracker=None,last_state_only: bool = True):#


        # print(his_features)
        # print(can_features.shape)
        sample_size=his_features.shape[0]
        his_features = self.dense(his_features)
        his_features = self.layer_norm(his_features)

        can_features = self.dense(can_features)
        can_features = self.layer_norm(can_features)


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

        

    def predict(self,his_features , can_features):

        sample_size=his_features.shape[0]
        his_features = self.dense(his_features)
        his_features = self.layer_norm(his_features)

        can_features = self.dense(can_features)
        can_features = self.layer_norm(can_features)


        res=torch.matmul(his_features,can_features.transpose(1,2))

        res=res.reshape(-1)
        #print('res: ',res)

        #res=F.sigmoid(res)
        #print('res: ',res)
        #print('res shape: ',res.shape)

        return res



















        










        




