import json
import pickle
import numpy as np
import random
# from fairseq.data import Dictionary
import sys
import torch
import argparse
import os
from model_dot import  Plain_bert
from fairseq.models.roberta import RobertaModel
from utils_sample_deepwalk import NewsIterator
from utils_sample_deepwalk import cal_metric
import utils_sample_deepwalk as utils
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
#import apex
random.seed(1)
np.random.seed(1) 
torch.manual_seed(1) 
torch.cuda.manual_seed(1)


cudaid=0
metrics=['group_auc','mean_mrr','ndcg@5;10']
lr=1e-4
T_warm=5000
all_iteration=33040


def parse_args():
    parser = argparse.ArgumentParser("Transformer-XH")

    parser.add_argument("--data_dir",
                    type=str,
                    help="local_rank for distributed training on gpus")
    
   
    
    parser.add_argument("--cudaid", "--cid",
                    help="pointer to the configuration file of the experiment", type=int, required=True)
    
    parser.add_argument("--save_dir",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--data_file",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--model_file",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--feature_file",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--log_file",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--batch_size",
                    type=int,
                    default=1,
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

def test(model, args):

    model.eval()
    test_file=os.path.join(args.data_dir, args.data_file)  
    cudaid=args.cudaid
    w=open(os.path.join(args.data_dir,args.log_file),'w')

    preds = []
    labels = []
    imp_indexes = []

    iterator=NewsIterator(batch_size=args.batch_size, npratio=-1,feature_file=os.path.join(args.data_dir,args.feature_file))
    #for epoch in range(0,100):
    batch_t=0
    iteration=0
    cudaid=args.cudaid
    
    #w=open(os.path.join(args.data_dir,args.log_file),'w')
    with torch.no_grad():
        data_batch=iterator.load_data_from_file(test_file)
        for  imp_index , his_id, candidate_id , label in data_batch:
            batch_t+=1
            
            his_id=his_id.cuda(cudaid)
            candidate_id= candidate_id.cuda(cudaid)
            logit=model.predict(his_id,candidate_id)

            logit=np.reshape(np.array(logit.cpu()), -1)
            label=np.reshape(np.array(label), -1)
            imp_index=np.reshape(np.array(imp_index), -1)
            #print('batch_t:',batch_t)
            for i in range(len(imp_index)):
                #print('imp_index: '+str(imp_index[i])+' logit: '+str(logit[i])+' label: '+str(label[i]))
                w.write('imp_index: '+str(imp_index[i])+' logit: '+str(logit[i])+' label: '+str(label[i])+'\n')
    w.close()

def exact_result3():
    preds = []
    labels = []
    imp_indexes = []
    preds_t = []
    labels_t = []
    imp_indexes_t = []
    x=1
    flag=''
    #for num in range(4):
        #f1=open('../data/res_transformer_xh_adduser3_'+str(num)+'_init.txt','r').readlines() 
    f1=open('../data/test_dot2.txt','r').readlines() 
    for line in f1:
        line=line.strip().split(' ')
        logit=float(line[3])
        imp_index=int(line[1])
        label=int(float(line[5]))
        labels.append(label)
        preds.append(logit)
        imp_indexes.append(imp_index)
    print('x: ',x)
    group_labels, group_preds = group_labels_func(labels, preds, imp_indexes)

    res = cal_metric(group_labels, group_preds, metrics)
    print(res)


if __name__ == '__main__':

    # cuda_num=int(sys.argv[1])
    exact_result3()
    assert 1==0

    random.seed(1)
    np.random.seed(1) 
    torch.manual_seed(1) 
    torch.cuda.manual_seed(1)
    #main()
    args = parse_args()
    model=Plain_bert()
    cudaid=args.cudaid

    #optimizer = apex.optimizers.FusedLAMB(model.parameters(), lr=lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0,max_grad_norm=1.0)
    model_dict = model.state_dict()
    # for k, v in model_dict.items(): 
    #     print(k,v.size())
    # print('----------------------------------------------------')
    #print(model.state_dict()['score3.0.bias'])

    # roberta = RobertaModel.from_pretrained('./model/roberta.base', checkpoint_file='model.pt')
    model_file=os.path.join(args.data_dir,args.model_file)
    save_model=torch.load(model_file, map_location=lambda storage, loc: storage)
    pretrained_dict={}
    #for name,parameters in roberta.named_parameters():
    for name in save_model:
    #   if name[7:] == "score3.0.bias":
    #       print(save_model[name])
        #print(name,':',save_model[name].size())
        #print(name,':',parameters.size())
        # if ( 'layers' in name ):
        #   pretrained_dict[name[31:]]=parameters
        # elif ('embed_positions.weight' in name or 'embed_tokens' in name or 'emb_layer_norm' in name):
        #   pretrained_dict[name[31:]]=parameters
        pretrained_dict[name[7:]]=save_model[name]
        # elif 'lm_head.' in name:
        #     pretrained_dict[name[14:]]=parameters
    # print('----------------------------------------------------------')
    print(pretrained_dict.keys())
    #assert 1==0
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda(cudaid)
    test(model,args)
    # for epoch in range(5):
    #   iteration=train(model,epoch,optimizer,iteration,cuda_num)
    #   res=test(model)
    #   print(res)
        # optimizer.step()
        # optimizer.zero_grad()

            
























