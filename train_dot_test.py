import json
import pickle
import numpy as np
import random
# from fairseq.data import Dictionary
import sys
import torch
import argparse
import os
from fairseq.models.roberta import RobertaModel
from utils_sample import NewsIterator
from utils_sample import cal_metric
import utils_sample as utils
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
random.seed(1)
np.random.seed(1) 
torch.manual_seed(1) 
torch.cuda.manual_seed_all(1)

from fairseq import utils as utils_fairseq

#cudaid=0
metrics=['group_auc','mean_mrr','ndcg@5;10']
lr=2e-5
T_warm=5000
all_iteration=50000


def parse_args():
    parser = argparse.ArgumentParser("Transformer-XH")

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
                    default=None,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--test_feature_file",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--size",
                    type=int,
                    default=1,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--gpu_size",
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
    parser.add_argument("--news_model_file",
                    type=str,
                    default=None,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--model_type",
                    type=str,
                    default='dot4',
                    help="local_rank for distributed training on gpus")


    # @staticmethod
    # def add_args(parser):
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
                        choices=utils_fairseq.get_available_activation_fns(),
                        help='activation function to use')
    parser.add_argument('--pooler-activation-fn',
                        choices=utils_fairseq.get_available_activation_fns(),
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


    parser.add_argument('--max_target_positions', type=int, metavar='N',default=512,
                        help='decoder output dimension (extra linear layer '
                             'if different from decoder embed dim')



    return parser.parse_args(),parser


def adjust_learning_rate(optimizer,iteration,lr=lr, T_warm=T_warm, all_iteration=all_iteration ):#得看一些一共有多少个iteration再确定
    if iteration<=T_warm:
        lr=lr*float(iteration)/T_warm
    elif iteration<all_iteration:
        lr = lr * (1 - (iteration - T_warm) / (all_iteration - T_warm))
    else:
        lr=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def group_labels_func(labels, preds, group_keys):
    """Devide labels and preds into several group according to values in group keys.

    Args:
        labels (list): ground truth label list.
        preds (list): prediction score list.
        group_keys (list): group key list.

    Returns:
        all_labels: labels after group.
        all_preds: preds after group.

    """

    all_keys = list(set(group_keys))
    group_labels = {k: [] for k in all_keys}
    group_preds = {k: [] for k in all_keys}
    
    for l, p, k in zip(labels, preds, group_keys):
        group_labels[k].append(l)
        group_preds[k].append(p)
    all_labels = []
    all_preds = []
    for k in all_keys:
        all_labels.append(group_labels[k])
        all_preds.append(group_preds[k])

    return all_labels, all_preds

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [data]

    world_size = dist.get_world_size()
    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def test(model,args,cudaid):
    preds = np.array([])
    labels = np.array([])
    imp_indexes = np.array([])
    metrics=['group_auc']
    test_file=os.path.join(args.data_dir, args.test_data_file)  
    preds = []
    labels = []
    imp_indexes = []
    if args.test_feature_file is not None:
        feature_file=os.path.join(args.data_dir,args.test_feature_file)
    else:
        feature_file=os.path.join(args.data_dir,args.feature_file)
    iterator=NewsIterator(batch_size=1, npratio=-1,feature_file=feature_file,field=args.field,fp16=True)
    print('test...')
    #cudaid=0
    #model = nn.DataParallel(model, device_ids=list(range(args.size)))
    step=0
    with torch.no_grad():
        data_batch=iterator.load_test_data_from_file(test_file,None,rank=cudaid,size=args.size)
        batch_t=0
        for  imp_index , user_index, his_id, candidate_id , label, _  in data_batch:
            batch_t+=len(candidate_id)
            his_id=his_id.cuda(cudaid)
            candidate_id= candidate_id.cuda(cudaid)
            logit=model(his_id,candidate_id,None,mode='validation')

            #print('???',his_id.shape,logit.shape,candidate_id.shape)
            # logit=list(np.reshape(np.array(logit.data.cpu()), -1))
            # label=list(np.reshape(np.array(label), -1))
            # imp_index=list(np.reshape(np.array(imp_index), -1))

            logit=np.reshape(np.array(logit.data.cpu()), -1)
            label=np.reshape(np.array(label), -1)
            #imp_index=np.reshape(np.array(imp_index), -1)

            assert len(imp_index)==1
            #imp_index=imp_index*len(logit)
            imp_index=np.repeat(imp_index,len(logit))

            assert len(logit)==len(label),(len(logit),len(label))
            assert len(logit)==len(imp_index)
            assert np.sum(label)!=0


            # labels.extend(label)
            # preds.extend(logit)
            # imp_indexes.extend(imp_index)
            labels=np.concatenate((labels,label),axis=0)
            preds=np.concatenate((preds,logit),axis=0)
            imp_indexes=np.concatenate((imp_indexes,imp_index),axis=0)
            step+=1
            if step%100==0:
                print('all data: ',len(labels),cudaid)
                #return labels,preds,imp_indexes

    # group_labels, group_preds = group_labels_func(labels, preds, imp_indexes)
    # res = cal_metric(group_labels, group_preds, metrics)
    # return res['group_auc']
    return labels,preds,imp_indexes

def train(cudaid, args,model):

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.size,
        rank=cudaid)

    random.seed(1)
    np.random.seed(1) 
    torch.manual_seed(1) 
    torch.cuda.manual_seed(1)

    print('params: '," T_warm: ",T_warm," all_iteration: ",all_iteration," lr: ",lr)
    #cuda_list=range(args.size)
    print('rank: ',cudaid)
    torch.cuda.set_device(cudaid)
    model.cuda(cudaid)

    accumulation_steps=int(args.batch_size/args.size/args.gpu_size)
    optimizer = apex.optimizers.FusedLAMB(model.parameters(), lr=lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0,max_grad_norm=1.0)
    model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    model = DDP(model)
    

    #model = nn.DataParallel(model, device_ids=cuda_list)
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    # torch.cuda.set_device(cudaid)
    
    #model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    #model=torch.nn.parallel.DistributedDataParallel(model, device_ids=cuda_list)
    #model = torch.nn.DataParallel(model)
    #model=apex.parallel.DistributedDataParallel(model)

    accum_batch_loss=0
    iterator=NewsIterator(batch_size=args.gpu_size, npratio=4,feature_file=os.path.join(args.data_dir,args.feature_file),field=args.field)
    train_file=os.path.join(args.data_dir, args.data_file)  
    #for epoch in range(0,100):
    batch_t=0
    iteration=0
    print('train...',args.field)
    #w=open(os.path.join(args.data_dir,args.log_file),'w')
    if cudaid==0:
        writer = SummaryWriter(os.path.join(args.save_dir, args.log_file) )
    epoch=0
    model.train()
    # batch_t=52880-1
    # iteration=3305-1
    batch_t=0
    iteration=0
    step=0
    best_score=-1
    #w=open(os.path.join(args.data_dir,args.log_file),'w')

    # model.eval()
    # auc=test(model,args)

    for epoch in range(0,200):
    #while True:
        all_loss=0
        all_batch=0
        data_batch=iterator.load_data_from_file(train_file,cudaid,args.size)
        for  imp_index , user_index, his_id, candidate_id , label in data_batch:
            batch_t+=1
            assert candidate_id.shape[1]==2
            his_id=his_id.cuda(cudaid)
            candidate_id= candidate_id.cuda(cudaid)
            label = label.cuda(cudaid)
            loss=model(his_id,candidate_id, label)

            sample_size=candidate_id.shape[0]
            loss=loss.sum()/sample_size/math.log(2)
            
            accum_batch_loss+=float(loss)

            all_loss+=float(loss)
            all_batch+=1

            loss = loss/accumulation_steps
            #loss.backward()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            if (batch_t)%accumulation_steps==0:

                iteration+=1
                adjust_learning_rate(optimizer,iteration)
                optimizer.step()
                optimizer.zero_grad()
                if cudaid==0:
                    print(' batch_t: ',batch_t, ' iteration: ', iteration, ' epoch: ',epoch,' accum_batch_loss: ',accum_batch_loss/accumulation_steps,' lr: ', optimizer.param_groups[0]['lr'])
                    writer.add_scalar('Loss/train', accum_batch_loss/accumulation_steps, iteration)
                    writer.add_scalar('Ltr/train', optimizer.param_groups[0]['lr'], iteration)
                accum_batch_loss=0
                if iteration%1000==0 :
                    torch.cuda.empty_cache()
                    model.eval()
                    labels,preds,imp_indexes = test(model,args,cudaid)
                    pred_pkl={'labels':labels,'preds':preds,'imp_indexes':imp_indexes}
                    all_preds=all_gather(pred_pkl)
                    if cudaid==0:
                        labels=np.concatenate([ele['labels'] for ele in all_preds], axis=0)
                        preds=np.concatenate([ele['preds'] for ele in all_preds], axis=0)
                        imp_indexes=np.concatenate([ele['imp_indexes'] for ele in all_preds], axis=0)
                        print('valid labels: ',len(labels))
                        group_labels, group_preds = group_labels_func(labels, preds, imp_indexes)
                        res = cal_metric(group_labels, group_preds, ['group_auc'])
                        auc = res['group_auc']
                        #auc=test(model,args)
                        print('valid auc: ', auc)
                        writer.add_scalar('valid/auc', auc, step)
                        step+=1
                        if auc>best_score:
                            torch.save(model.state_dict(), os.path.join(args.save_dir,'Plain_robert_dot_best.pkl'))
                            best_score=auc
                            print('best score: ',best_score)
                        torch.save(model.state_dict(), os.path.join(args.save_dir,'Plain_robert_dot_'+str(iteration)+'.pkl'))
                    torch.cuda.empty_cache()
                    model.train()
                if iteration >=all_iteration:
                    break
        
        if iteration >=all_iteration:
            break


                    
        # if cudaid==0:
        #     torch.save(model.state_dict(), os.path.join(args.save_dir,'Plain_robert_dot'+str(epoch)+'.pkl'))
    #w.close()
            

if __name__ == '__main__':

    # cuda_num=int(sys.argv[1])
    random.seed(1)
    np.random.seed(1) 
    torch.manual_seed(1) 
    torch.cuda.manual_seed(1)
    #main()
    args,parser = parse_args()
    if args.model_type=='dot4':
        from model_plain_bert_dot4_test import  Plain_bert
    elif args.model_type=='twotower_dot4':
        from model_twotower_bert_dot4 import  Plain_bert
    model=Plain_bert(args,parser)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0)
    
    # for name, param in model.named_parameters():
    #     print(name,param.shape,param.requires_grad)

    #roberta = RobertaModel.from_pretrained(os.path.join(args.data_dir,'roberta.base'), checkpoint_file='model.pt')
    # roberta = RobertaModel.from_pretrained(os.path.join(args.data_dir,'roberta.base'), checkpoint_file=args.model_file)
    # model_dict = model.state_dict()
    # pretrained_dict={}
    # for name,parameters in roberta.named_parameters():
    #     if  'lm_head' not in name:
    #         pretrained_dict['encoder.'+name[31:]]=parameters



    #finetune my rroduced roberta
    model_dict = model.state_dict()
    print('load: ',args.model_file)
    save_model=torch.load(args.model_file, map_location=lambda storage, loc: storage)
    #print(save_model['model'].keys())
    pretrained_dict= {}
    #print('???',save_model['model'].keys())
    for name in save_model['model']:
        if 'lm_head' not in name and 'encoder' in name and 'decode' not in name:
            pretrained_dict['encoder'+name[24:]]=save_model['model'][name]
        elif 'decode' in name:
            pretrained_dict[name]=save_model['model'][name]
    if args.news_model_file is not None:
        print('load: ',args.news_model_file)
        save_model2=torch.load(args.news_model_file, map_location=lambda storage, loc: storage)
        for name in save_model2['model']:
            if 'lm_head' not in name and 'encoder' in name and 'decode' not in name:
                pretrained_dict['news_encoder'+name[24:]]=save_model2['model'][name]

    if 'twotower' not in args.model_type:
        assert len(model_dict)-4==len(pretrained_dict), (len(model_dict),len(pretrained_dict),model_dict.keys(),pretrained_dict.keys())
    else:
        assert len(model_dict)-8==len(pretrained_dict), (len(model_dict),len(pretrained_dict),model_dict.keys(),pretrained_dict.keys())

    print(pretrained_dict.keys(),len(pretrained_dict.keys()))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    args.world_size = args.size * 1
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=args.size, args=(args,model))


    # model.cuda(cudaid)
    # train(model,optimizer,args)
    

            
























