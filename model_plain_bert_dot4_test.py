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
from fairseq.models.fairseq_encoder import EncoderOut

from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.roberta import RobertaModel
import torch
import torch.optim as optim
from fairseq.data import Dictionary
import numpy as np
#from fairseq.models.roberta import RobertaModel
from fairseq.modules import (
    LayerNorm,
    TransformerSentenceEncoder,
)
from fairseq.models.transformer import TransformerDecoder
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


def add_args(parser):
    """Add model-specific arguments to the parser."""
    parser.add_argument('--encoder-layers', type=int, metavar='L',
                        help='num encoder layers')
    parser.add_argument('--encoder-embed-dim', type=int, metavar='H',
                        help='encoder embedding dimension')
    parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='F',
                        help='encoder embedding dimension for FFN')
    parser.add_argument('--encoder-attention-heads', type=int, metavar='A',
                        help='num encoder attention heads')
    parser.add_argument('--activation-fn',
                        choices=utils.get_available_activation_fns(),
                        help='activation function to use')
    parser.add_argument('--pooler-activation-fn',
                        choices=utils.get_available_activation_fns(),
                        help='activation function to use for pooler layer')
    parser.add_argument('--encoder-normalize-before', action='store_true',
                        help='apply layernorm before each encoder block')
    parser.add_argument('--dropout', type=float, metavar='D',
                        help='dropout probability')
    parser.add_argument('--attention-dropout', type=float, metavar='D',
                        help='dropout probability for attention weights')
    parser.add_argument('--activation-dropout', type=float, metavar='D',
                        help='dropout probability after activation in FFN')
    parser.add_argument('--pooler-dropout', type=float, metavar='D',
                        help='dropout probability in the masked_lm pooler layers')
    parser.add_argument('--max-positions', type=int,
                        help='number of positional embeddings to learn')
    parser.add_argument('--load-checkpoint-heads', action='store_true',
                        help='(re-)register and load heads when loading checkpoints')

    parser.add_argument('--not-load-checkpoint-decoder', action='store_true',
                        help='do not load the decoder')
    # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                        help='LayerDrop probability for encoder')
    parser.add_argument('--encoder-layers-to-keep', default=None,
                        help='which layers to *keep* when pruning as a comma-separated list')
    # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                        help='iterative PQ quantization noise at training time')
    parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                        help='block size of quantization noise at training time')
    parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                        help='scalar quantization noise and scalar quantization at training time')
    parser.add_argument('--untie-weights-roberta', action='store_true',
                        help='Untie weights between embeddings and classifiers in RoBERTa')

    parser.add_argument('--train-ratio', type=str, metavar='STR',
                        help='loss train-ratio')

    parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                        help='path to pre-trained decoder embedding')
    parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                        help='decoder embedding dimension')
    parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                        help='decoder embedding dimension for FFN')
    parser.add_argument('--decoder-layers', type=int, metavar='N',
                        help='num decoder layers')
    parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                        help='num decoder attention heads')
    parser.add_argument('--decoder-learned-pos', action='store_true',
                        help='use learned positional embeddings in the decoder')
    parser.add_argument('--decoder-normalize-before', action='store_true',
                        help='apply layernorm before each decoder block')
    parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                        help='decoder output dimension (extra linear layer '
                             'if different from decoder embed dim')
    parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                        help='share decoder input and output embeddings')
    parser.add_argument('--share-all-embeddings', action='store_true',
                        help='share encoder, decoder and output embeddings'
                             ' (requires shared dictionary and embed dim)')
    parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                        help='if set, disables positional embeddings (outside self attention)')
    parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                        help='comma separated list of adaptive softmax cutoff points. '
                             'Must be used with adaptive_loss criterion'),
    parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                        help='sets adaptive softmax dropout for the tail projections')
    parser.add_argument('--layernorm-embedding', action='store_true',
                        help='add layernorm to embedding')
    parser.add_argument('--no-scale-embedding', action='store_true',
                        help='if True, dont scale embeddings')
    # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
    parser.add_argument('--no-cross-attention', default=False, action='store_true',
                        help='do not perform cross-attention')
    parser.add_argument('--cross-self-attention', default=False, action='store_true',
                        help='perform cross+self-attention')
    # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    
    parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                        help='LayerDrop probability for decoder')
    parser.add_argument('--decoder-layers-to-keep', default=None,
                        help='which layers to *keep* when pruning as a comma-separated list')


    parser.add_argument('--decoder-atten-window', type=int, metavar='N',default=16,
                        help='decoder output dimension (extra linear layer '
                             'if different from decoder embed dim')

class Plain_bert(nn.Module):#
    def __init__(self,args,parser):
        super().__init__()
        embedding_dim=768
        roberta_decoder_layer3_architecture(args)
        #add_args(parser)
        print('???',args.decoder_embed_dim)
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
        source_dictionary = Dictionary.load(os.path.join('/home/dihe/Projects/data/bert-16g-0930', 'dict.txt'))
        self.decoder=TransformerDecoder(args, source_dictionary, self.encoder.embed_tokens,no_encoder_attn=getattr(args, "no_cross_attention", False))


    

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
        his_features_ori=his_features
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

        his_decode=torch.cat((his_id[:,1:],torch.ones(batch_size,1).cuda()),dim=1)
        h=EncoderOut(
                encoder_out=his_features_ori.transpose(0,1),  # T x B x C
                encoder_padding_mask=None,  # B x T
                encoder_embedding=None,  # B x T x C
                encoder_states=None,  # List[T x B x C]
                src_tokens=None,
                src_lengths=None,
            )
        decoder_output=self.decoder(his_decode,encoder_out=h,local_attn_mask=2)[0]
        decode_loss = modules.cross_entropy(
            decoder_output.view(-1, decoder_output.size(-1)),
            his_id.view(-1),
            reduction='mean',
            ignore_index=1,
        )


        print('decode loss: ',decode_loss)
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

def base_architecture(args):
    setattr(args, 'encoder_layers', 12)
    setattr(args, 'encoder_embed_dim', 768)
    setattr(args, 'encoder_ffn_embed_dim', 3072)
    setattr(args, 'encoder_attention_heads', 12)

    setattr(args, 'activation_fn', 'gelu')
    setattr(args, 'pooler_activation_fn', 'tanh')

    setattr(args, 'dropout', 0.1)
    setattr(args, 'attention_dropout', 0.1)
    setattr(args, 'activation_dropout', 0.0)
    setattr(args, 'pooler_dropout', 0.0)
    setattr(args, 'encoder_layers_to_keep', None)
    setattr(args, 'encoder_layerdrop', 0.0)


    
    setattr(args, "decoder_embed_path", None)
    setattr(args, "decoder_embed_dim", 768)
    setattr(
        args, "decoder_ffn_embed_dim", 3072
    )
    #setattr(args, "decoder_layers", 12)
    setattr(args, "decoder_attention_heads", 12)
    setattr(args, "decoder_normalize_before", True)
    setattr(args, "decoder_learned_pos", True)
    setattr(args, "attention_dropout", 0.1)
    setattr(args, "activation_dropout", 0.0)
    setattr(args, "activation_fn", "relu")
    setattr(args, "dropout", 0.1)
    setattr(args, "adaptive_softmax_cutoff", None)
    setattr(args, "adaptive_softmax_dropout", 0)
    setattr(
        args, "share_decoder_input_output_embed", True
    )
    setattr(args, "share_all_embeddings", True)
    setattr(
        args, "no_token_positional_embeddings", False
    )
    setattr(args, "adaptive_input", False)
    setattr(args, "no_cross_attention", False)
    setattr(args, "cross_self_attention", False)

    setattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    setattr(args, "decoder_input_dim", args.decoder_embed_dim)

    setattr(args, "no_scale_embedding", True)
    setattr(args, "layernorm_embedding", True)
    setattr(args, "tie_adaptive_weights", True)#不确定啊

    print('!!!',args.decoder_embed_dim)





def roberta_decoder_layer3_architecture(args):
    setattr(args, 'decoder_layers', 3)
    base_architecture(args)



















        










        




