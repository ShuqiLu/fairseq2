import json
import pickle
import numpy as np
import random
# from fairseq.data import Dictionary
import sys
import torch
import argparse
import os
# from model_plain_bert_hf import  Plain_bert
from model_sparse_bert_hf import  Plain_bert
# from fairseq.models.roberta import RobertaModel
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
# from fairseq.data import (
#     data_utils,
#     Dictionary,
#     IdDataset,
#     MaskTokensDataset,
#     NestedDictionaryDataset,
#     NumelDataset,
#     NumSamplesDataset,
#     PadDataset,
#     PrependTokenDataset,
#     SortDataset,
#     TokenBlockDataset,
# )
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
T_warm=5000
all_iteration=41316


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
    parser.add_argument("--history_file",
                    type=str,
                    help="local_rank for distributed training on gpus")

    parser.add_argument("--abs_file",
                    type=str,
                    help="local_rank for distributed training on gpus")



    return parser.parse_args()


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



def test(model,args):
    preds = []
    labels = []
    imp_indexes = []
    metrics=['group_auc']
    test_file=os.path.join(args.data_dir, args.test_data_file)  
    preds = []
    labels = []
    imp_indexes = []
    feature_file=os.path.join(args.data_dir,args.feature_file)
    history_file=os.path.join(args.data_dir,args.history_file)
    if 'last' in args.field:
        abs_file=os.path.join(args.data_dir,args.abs_file)
    else:
        abs_file=''
    iterator=NewsIterator(batch_size=1, npratio=-1,feature_file=feature_file,history_file=history_file,abs_file=abs_file,field=args.field,fp16=True)
    print('test...')
    cudaid=0
    #model = nn.DataParallel(model, device_ids=list(range(args.size)))
    step=0
    with torch.no_grad():
        data_batch=iterator.load_test_data_from_file(test_file,None)
        batch_t=0
        for  imp_index , user_index, his_id, candidate_id , label, _  in data_batch:
            batch_t+=len(candidate_id)
            his_id=his_id.cuda(cudaid)
            candidate_id= candidate_id.cuda(cudaid)
            logit=model(his_id,candidate_id,None,mode='validation')
            # print('???',label_t,label)
            # assert 1==0
            logit=list(np.reshape(np.array(logit.cpu()), -1))
            label=list(np.reshape(np.array(label), -1))
            imp_index=list(np.reshape(np.array(imp_index), -1))

            assert len(imp_index)==1
            imp_index=imp_index*len(logit)

            assert len(logit)==len(label)
            assert len(logit)==len(imp_index)
            assert np.sum(np.array(label))!=0

            labels.extend(label)
            preds.extend(logit)
            imp_indexes.extend(imp_index)
            step+=1
            if step%100==0:
                print('all data: ',len(labels))

    group_labels, group_preds = group_labels_func(labels, preds, imp_indexes)
    res = cal_metric(group_labels, group_preds, metrics)
    return res['group_auc']

def train(cudaid, args,model):
    pynvml.nvmlInit()
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
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0)
    optimizer = apex.optimizers.FusedLAMB(model.parameters(), lr=lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0,max_grad_norm=1.0)
    
    model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    
    model = DDP(model)
    
    accum_batch_loss=0
    history_file=os.path.join(args.data_dir,args.history_file)
    if 'last' in args.field:
        abs_file=os.path.join(args.data_dir,args.abs_file)
    else:
        abs_file=''
    iterator=NewsIterator(batch_size=args.gpu_size, npratio=4,feature_file=os.path.join(args.data_dir,args.feature_file),history_file=history_file,abs_file=abs_file,field=args.field,fp16=True)
    train_file=os.path.join(args.data_dir, args.data_file)  
    batch_t=0
    iteration=0
    print('train...',args.field)
    if cudaid==0:
        writer = SummaryWriter(os.path.join(args.data_dir, args.log_file) )
    epoch=0
    model.train()
    batch_t=0
    iteration=0
    step=0
    best_score=-1

    for epoch in range(0,10):
        all_loss=0
        all_batch=0
        data_batch=iterator.load_data_from_file(train_file,cudaid,args.size)
        print('load ok...')
        for  imp_index , user_index, his_id, candidate_id , label in data_batch:
            batch_t+=1
            assert candidate_id.shape[1]==2

            # if cudaid==1:
            #     torch.set_printoptions(profile="full")
            #     print(his_id)
            his_id=his_id.cuda(cudaid)
            candidate_id= candidate_id.cuda(cudaid)
            label = label.cuda(cudaid)
            loss=model(his_id,candidate_id, label)

            sample_size=candidate_id.shape[0]
            loss=loss.sum()/sample_size/math.log(2)
            
            accum_batch_loss+=float(loss)

            all_loss+=float(loss)
            all_batch+=1

            # if cudaid==1:
                
                # torch.set_printoptions(profile="full")
                # w=open('input.txt','w')
                # w.write(str(his_id.cpu()))
                # w.close()
                # assert 1==0

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
                    # handle = pynvml.nvmlDeviceGetHandleByIndex(cudaid)
                    # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    # #print(int(meminfo.used)/1024/1024)
                    # print('loss: ',loss,int(meminfo.used)/1024/1024)
                    print(' batch_t: ',batch_t, ' iteration: ', iteration, ' epoch: ',epoch,' accum_batch_loss: ',accum_batch_loss/accumulation_steps,' lr: ', optimizer.param_groups[0]['lr'])
                    writer.add_scalar('Loss/train', accum_batch_loss/accumulation_steps, iteration)
                    writer.add_scalar('Ltr/train', optimizer.param_groups[0]['lr'], iteration)
                accum_batch_loss=0
                if iteration%500==0 and cudaid==0:
                    torch.cuda.empty_cache()
                    model.eval()
                    if cudaid==0:
                        auc=test(model,args)
                        print(auc)
                        writer.add_scalar('auc/valid', auc, step)
                        step+=1
                        if auc>best_score:
                            torch.save(model.state_dict(), os.path.join(args.save_dir,'Plain_robert_dot_best.pkl'))
                            best_score=auc
                            print('best score: ',best_score)
                    torch.cuda.empty_cache()
                    model.train()
        
        if cudaid==0:
            torch.save(model.state_dict(), os.path.join(args.save_dir,'Plain_robert_dot'+str(epoch)+'.pkl'))            

if __name__ == '__main__':

    # cuda_num=int(sys.argv[1])
    random.seed(1)
    np.random.seed(1) 
    torch.manual_seed(1) 
    torch.cuda.manual_seed(1)
    #main()
    args = parse_args()
    model=Plain_bert(args)

    #optimizer = torch.optim.Adam(model.parameters(), lr=lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0)
    
    # for name, param in model.named_parameters():
    #     print(name,param.shape,param.requires_grad,param)
    #     break

    # roberta = RobertaModel.from_pretrained(os.path.join(args.data_dir,'roberta.base'), checkpoint_file='model.pt')
    # #roberta = RobertaModel.from_pretrained(os.path.join(args.data_dir,'roberta.base'), checkpoint_file='checkpoint_best.pt')
    # for name, param in roberta.named_parameters():
    #     print(name,param.shape,param.requires_grad,param)
    #     break
    # assert 1==0

    # model_dict = model.state_dict()
    # pretrained_dict={}
    # for name,parameters in roberta.named_parameters():
    #     if  'lm_head' not in name:
    #         pretrained_dict['encoder.'+name[31:]]=parameters

    # print(pretrained_dict.keys(),len(pretrained_dict.keys()))
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    args.world_size = args.size * 1
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=args.size, args=(args,model))

    # train(0,args,model)
    

            
























