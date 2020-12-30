import json
import pickle
import numpy as np
import random
# from fairseq.data import Dictionary
import sys
import torch
import argparse
import os
from model_pretrain_mlm import  Plain_bert
from fairseq.models.roberta import RobertaModel
# from utils_sample import NewsIterator
# from utils_sample import cal_metric
from fairseq import utils as fairseq_utils
import utils_pretrain_mlm as utils
# import dgl
# import dgl.function as fn
#from gpu_mem_track import  MemTracker
#import inspect
#from multiprocessing import Pool
import torch.nn as nn
import math
from fairseq.data import (
    data_utils,
    Dictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PadDataset,
    PrependTokenDataset,
    SortDataset,
    TokenBlockDataset,
)
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from apex.parallel import DistributedDataParallel as DDP
import apex
from apex import amp
import torch.multiprocessing as mp
import torch.distributed as dist
import pynvml
random.seed(1)
np.random.seed(1) 
torch.manual_seed(1) 
torch.cuda.manual_seed(1)


#cudaid=0
metrics=['group_auc','mean_mrr','ndcg@5;10']
lr=1e-4
T_warm=10000
all_iteration=1000000


def parse_args(parser):
    

    parser.add_argument("--data_dir",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--save_dir",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--data_file",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--test_data_file",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--feature_file",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--test_feature_file",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--world_size",
                    type=int,
                    default=1,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--gpu_size",
                    type=int,
                    default=1,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--valid_size",
                    type=int,
                    default=1,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--batch_size",
                    type=int,
                    default=1,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--log_file",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--field",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--model_file",
                    type=str,
                    help="local_rank for distributed training on gpus")

    parser.add_argument("--batch_t",
                    type=int,
                    default=1,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--iteration",
                    type=int,
                    default=1,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--epoch",
                    type=int,
                    default=1,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--batch_one_epoch",
                    type=int,
                    help="local_rank for distributed training on gpus")
    parser.add_argument('--use_start_pos', action='store_true',
                        help='apply layernorm before each encoder block')
    parser.add_argument('--from_epoch', action='store_true',
                        help='apply layernorm before each encoder block')

    parser.add_argument("--all_batch_loss",
                    type=float,
                    help="local_rank for distributed training on gpus")


#     return parser.parse_args()



# def parse_args_model(parser):
    parser.add_argument('--activation-fn',
                            choices=fairseq_utils.get_available_activation_fns(),
                            help='activation function to use')
    parser.add_argument('--dropout', type=float, metavar='D',
                        help='dropout probability')
    parser.add_argument('--attention-dropout', type=float, metavar='D',
                        help='dropout probability for attention weights')
    parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                        help='dropout probability after activation in FFN.')
    parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                        help='path to pre-trained encoder embedding')
    parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                        help='encoder embedding dimension')
    parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                        help='encoder embedding dimension for FFN')
    parser.add_argument('--encoder-layers', type=int, metavar='N',
                        help='num encoder layers')
    parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                        help='num encoder attention heads')
    parser.add_argument('--encoder-normalize-before', action='store_true',
                        help='apply layernorm before each encoder block')
    parser.add_argument('--encoder-learned-pos', action='store_true',
                        help='use learned positional embeddings in the encoder')
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
    parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                        help='LayerDrop probability for encoder')
    parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                        help='LayerDrop probability for decoder')
    parser.add_argument('--encoder-layers-to-keep', default=None,
                        help='which layers to *keep* when pruning as a comma-separated list')
    parser.add_argument('--decoder-layers-to-keep', default=None,
                        help='which layers to *keep* when pruning as a comma-separated list')
    # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                        help='iterative PQ quantization noise at training time')
    parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                        help='block size of quantization noise at training time')
    parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                        help='scalar quantization noise and scalar quantization at training time')



    return parser.parse_args()

def base_architecture(args):
    #dropout 不确定
    #decoder_layerdrop 不确定
    #share_decoder_input_output_embed ok
    #decoder_embed_dim 不确定
    #decoder_output_dim 不确定
    #max_target_positions
    #no_scale_embedding

    #adaptive_input
    #quant_noise_pq
    #quant_noise_pq
    #quant_noise_pq_block_size


    #decoder_learned_pos 不确定
    #no_token_positional_embeddings 不确定
    #decoder_layers 不确定
    #decoder_normalize_before 不确定但感觉应该是True


    #tie_adaptive_weights
    #adaptive_softmax_cutoff
    #adaptive_softmax_dropout
    #adaptive_softmax_factor
    #tie_adaptive_proj


    setattr(args, "encoder_embed_path", None)
    setattr(args, "encoder_embed_dim", 768)
    setattr(args, "encoder_ffn_embed_dim", 3072)
    setattr(args, "encoder_layers", 12)
    setattr(args, "encoder_attention_heads", 12)
    setattr(args, "encoder_normalize_before", True)
    setattr(args, "encoder_learned_pos", True)
    setattr(args, "decoder_embed_path", None)
    setattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    setattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    setattr(args, "decoder_layers", 12)
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

    print('???',args.encoder_embed_dim)


def adjust_learning_rate(optimizer,iteration,lr=lr, T_warm=T_warm, all_iteration=all_iteration ):#得看一些一共有多少个iteration再确定
    if iteration<=T_warm:
        lr=lr*float(iteration)/T_warm
    elif iteration<all_iteration:
        lr = lr * (1 - (iteration - T_warm) / (all_iteration - T_warm))
    else:
        lr=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def test(model,args,mlm_data,roberta_dict,rerank):
    
    print('test...')
    cudaid=0
    #model = nn.DataParallel(model, device_ids=list(range(args.world_size)))
    step=0
    accum_batch_loss=0
    accum_batch_loss_decode=0
    accum_batch_loss_mask=0
    accumulation_steps=0

    batch_t=0

    with torch.no_grad():
        data_batch=utils.get_batch(mlm_data,roberta_dict,args.valid_size,rerank=rerank,mode='valid')
        for  token_list, mask_label_list in data_batch:
            #batch_t+=1
            #assert candidate_id.shape[1]==2
            # his_id=his_id.cuda(cudaid)
            # candidate_id= candidate_id.cuda(cudaid)
            # label = label.cuda(cudaid)
            # loss=model(his_id,candidate_id, label)
            batch_t+=token_list.shape[0]

            token_list=token_list.cuda(cudaid)
            mask_label_list=mask_label_list.cuda(cudaid)

            loss_mask,sample_size_mask=model(token_list,mask_label_list)

            loss_mask=loss_mask/sample_size_mask/math.log(2)
            loss=loss_mask

            # print('loss: ',loss,' sample_size: ',sample_size)
            # assert 1==0
            
            accum_batch_loss+=float(loss)

            accumulation_steps+=1

            if accumulation_steps%100==0:
                print('batch_t: ',batch_t)


    return accum_batch_loss/accumulation_steps

def train(cudaid, args,model,roberta_dict,rerank):

    #pynvml.nvmlInit()
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=cudaid)


    random.seed(1)
    np.random.seed(1) 
    torch.manual_seed(1) 
    torch.cuda.manual_seed(1)

    print('params: '," T_warm: ",T_warm," all_iteration: ",all_iteration," lr: ",lr)
    #cuda_list=range(args.world_size)
    print('rank: ',cudaid)
    torch.cuda.set_device(cudaid)
    model.cuda(cudaid)

    accumulation_steps=int(args.batch_size/args.world_size/args.gpu_size)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0)

    #optimizer = apex.optimizers.FusedLAMB(model.parameters(), lr=lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0,max_grad_norm=1.0)
    optimizer = apex.optimizers.FusedLAMB(model.parameters(), lr=lr,betas=(0.9,0.999),eps=1e-6,weight_decay=0.01,max_grad_norm=1.0)
    model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    model = DDP(model)
    mlm_data=utils.load_mask_data(os.path.join(args.data_dir,'data-bin-body1_3/train' ),roberta_dict)
    #decode_data=utils.load_decode_data(os.path.join(args.data_dir,'data-bin-abs1_3/train'),roberta_dict)


    #model = nn.DataParallel(model, device_ids=cuda_list)
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    # torch.cuda.set_device(cudaid)
    
    #model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    #model=torch.nn.parallel.DistributedDataParallel(model, device_ids=cuda_list)
    #model = torch.nn.DataParallel(model)
    #model=apex.parallel.DistributedDataParallel(model)

    accum_batch_loss=0
    accum_batch_loss_decode=0
    accum_batch_loss_mask=0

    all_batch_loss=0
    all_batch_loss_decode=0
    all_batch_loss_mask=0
    #iterator=NewsIterator(batch_size=args.gpu_size, npratio=4,feature_file=os.path.join(args.data_dir,args.feature_file),field=args.field)
    #train_file=os.path.join(args.data_dir, args.data_file)  
    #for epoch in range(0,100):
    batch_t=0
    iteration=0
    print('train...',args.field)
    #w=open(os.path.join(args.data_dir,args.log_file),'w')
    if cudaid==0:
        writer = SummaryWriter(os.path.join(args.data_dir, args.log_file) )
    epoch=0
    model.train()
    # batch_t=52880-1
    # iteration=3305-1
    batch_t=0
    iteration=0
    step=0
    best_score=-1
    step_t=0

    start_pos=None
    batch_t_arg=-1
    #w=open(os.path.join(args.data_dir,args.log_file),'w')

    # model.eval()
    # auc=test(model,args)
    if args.model_file !=None:
        epoch_o=args.epoch
        iteration=args.iteration
        #batch_t=args.batch_t
        step=int(iteration/10000)+1
        if args.use_start_pos:
            start_pos=args.gpu_size*batch_t*2%(int((32255176-int(0.002*32255176))/args.world_size)+1)
            batch_t_arg=args.batch_t
            batch_t=args.batch_t
        elif args.from_epoch:
            batch_t_arg=args.batch_t
            batch_t=args.batch_t
        elif args.batch_one_epoch!=None:
            batch_t_arg=args.batch_t%args.batch_one_epoch
        else:
            batch_t_arg=args.batch_t

        all_batch_loss=args.all_batch_loss * args.batch_t



    #w=open(os.path.join(args.data_dir,args.log_file),'w')

    # model.eval()
    # auc=test(model,args)

    for epoch in range(epoch_o,20):
    #while True:
        #data_batch=iterator.load_data_from_file(train_file,cudaid,args.world_size)
        data_batch=utils.get_batch(mlm_data,roberta_dict,args.gpu_size,rerank=rerank,mode='train',dist=True,cudaid=cudaid,size=args.world_size,start_pos=start_pos)
        start_pos=None
        for  token_list, mask_label_list in data_batch:
            if epoch==epoch_o and batch_t<batch_t_arg:
                batch_t+=1
                continue
            batch_t+=1
            #assert candidate_id.shape[1]==2
            # his_id=his_id.cuda(cudaid)
            # candidate_id= candidate_id.cuda(cudaid)
            # label = label.cuda(cudaid)
            # loss=model(his_id,candidate_id, label)

            token_list=token_list.cuda(cudaid)
            mask_label_list=mask_label_list.cuda(cudaid)

            loss_mask,sample_size_mask=model(token_list,mask_label_list)

            #print('????decode: ',sample_size_decode)
            #print('output: ',loss_mask,sample_size_mask,loss_decode,sample_size_decode)

            if sample_size_mask!=0:
                loss_mask=loss_mask/sample_size_mask/math.log(2)

            
            loss=loss_mask
            #loss=loss_decode
            # print('loss: ',loss,' sample_size: ',sample_size)
            # assert 1==0

            # if cudaid==0:
            #     print('shape: ',token_list.shape,' batch_t: ',batch_t,' loss: ',loss,' loss_mask: ',loss_mask,' loss_decode: ',loss_decode)
            
            accum_batch_loss+=float(loss)
            

            all_batch_loss+=float(loss)
            

            loss = loss/accumulation_steps

            # loss.backward()

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            if (batch_t)%accumulation_steps==0:

                # handle = pynvml.nvmlDeviceGetHandleByIndex(cudaid)
                # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                # #print(int(meminfo.used)/1024/1024)
                # print('memory: ',int(meminfo.used)/1024/1024,' cudaid: ',cudaid)

                iteration+=1
                adjust_learning_rate(optimizer,iteration)
                optimizer.step()
                optimizer.zero_grad()
                if cudaid==0:
                    print(' batch_t: ',batch_t, ' iteration: ', iteration, ' epoch: ',epoch,' accum_batch_loss: ',accum_batch_loss/accumulation_steps, ' all_batch_loss: ', all_batch_loss/batch_t , \
                        ' lr: ', optimizer.param_groups[0]['lr'])
                    writer.add_scalar('Loss/train', accum_batch_loss/accumulation_steps, iteration)
                    

                    writer.add_scalar('Loss_all/train', all_batch_loss/batch_t, iteration)

                    writer.add_scalar('Ltr/train', optimizer.param_groups[0]['lr'], iteration)

                accum_batch_loss=0
                accum_batch_loss_mask=0
                accum_batch_loss_decode=0

                if iteration%5000==0 and cudaid==0:
                    torch.cuda.empty_cache()
                    model.eval()
                    if cudaid==0:
                        accum_batch_loss_t=test(model,args,mlm_data,roberta_dict,rerank)
                        print('valid loss: ',accum_batch_loss_t)
                        writer.add_scalar('Loss/valid', accum_batch_loss_t, step)
                        torch.save(model.state_dict(), os.path.join(args.save_dir,'pretrain_iteration'+str(iteration)+'.pkl'))
                        step+=1
                        # if auc>best_score:
                        #     torch.save(model.state_dict(), os.path.join(args.save_dir,'pretrain_best.pkl'))
                        #     best_score=auc
                        #     print('best score: ',best_score)
                    torch.cuda.empty_cache()
                    model.train()

        if cudaid==0:
            torch.save(model.state_dict(), os.path.join(args.save_dir,'doc_roberta'+str(epoch)+'.pkl'))
    #w.close()
            

if __name__ == '__main__':

    # cuda_num=int(sys.argv[1])
    random.seed(1)
    np.random.seed(1) 
    torch.manual_seed(1) 
    torch.cuda.manual_seed(1)
    #main()
    parser = argparse.ArgumentParser("pretraining-model")
    args = parse_args(parser)
    #args=parse_args_model(parser)
    base_architecture(args)
    print('???',args.encoder_embed_dim)

    #roberta_dict=utils.load_dict(args.data_dir)
    roberta_dict=Dictionary.load(os.path.join(args.data_dir, 'roberta.base/dict.txt') )

    model=Plain_bert(args,roberta_dict)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0)
    
    # for name, param in model.named_parameters():
    #     print(name,param.shape,param.requires_grad)

    #roberta = RobertaModel.from_pretrained(os.path.join(args.data_dir,'roberta.base'), checkpoint_file='model.pt')
    #roberta = RobertaModel.from_pretrained(os.path.join(args.data_dir,'roberta.base'), checkpoint_file=args.model_file)

    # for name, param in roberta.named_parameters():
    #     print(name,param.shape,param.requires_grad)


    # model_dict = model.state_dict()
    # pretrained_dict={}
    # for name,parameters in roberta.named_parameters():
    #     if  'lm_head' not in name:
    #         pretrained_dict['encoder.'+name[31:]]=parameters

    # print(pretrained_dict.keys(),len(pretrained_dict.keys()))
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    if args.model_file !=None:
        model_dict = model.state_dict()
        model_file=os.path.join(args.save_dir,args.model_file)
        save_model=torch.load(model_file, map_location=lambda storage, loc: storage)
        pretrained_dict={}
        for name in save_model:      
            pretrained_dict[name[7:]]=save_model[name]        
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)


    if os.path.exists(os.path.join(args.data_dir,'rerank1_3.npy')): 
        rerank=np.load(os.path.join(args.data_dir,'rerank1_3.npy'))
    else:
        rerank=np.arange(32255176)
        random.shuffle(rerank)
        np.save(os.path.join(args.data_dir,'rerank1_3.npy'),rerank)

    print('rerank: ',rerank[:10])

    args.world_size = args.world_size * 1
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=args.world_size, args=(args,model,roberta_dict,rerank))

    # model.cuda(cudaid)
    
    #train(0,args,model,roberta_dict,rerank)
    

            
























