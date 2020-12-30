import json
import pickle
import numpy as np
import random
# from fairseq.data import Dictionary
import sys
import torch
import argparse
import os
# from model_plain_bert_dot2 import  Plain_bert
from model_plain_bert_hf import  Plain_bert
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
import apex
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
    parser.add_argument("--batch_size",
                    type=int,
                    default=1,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--gpu_size",
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

def test(model,arges):
    preds = []
    labels = []
    imp_indexes = []
    metrics=['group_auc']
    test_file=os.path.join(args.data_dir, args.test_data_file)  
    preds = []
    labels = []
    imp_indexes = []
    feature_file=os.path.join(args.data_dir,args.feature_file)
    iterator=NewsIterator(batch_size=900, npratio=-1,feature_file=feature_file,field=args.field)
    print('test...')
    with torch.no_grad():
        data_batch=iterator.load_data_from_file(test_file)
        batch_t=0
        for  imp_index , user_index, his_id, candidate_id , label  in data_batch:
            batch_t+=len(candidate_id)
            his_id=his_id.cuda(cudaid)
            candidate_id= candidate_id.cuda(cudaid)
            logit=model(his_id,candidate_id,None,mode='validation')
            # print('???',label_t,label)
            # assert 1==0
            logit=list(np.reshape(np.array(logit.cpu()), -1))
            label=list(np.reshape(np.array(label), -1))
            imp_index=list(np.reshape(np.array(imp_index), -1))
            #print('???',len(logit),len(label))
            # for i in range(len(label)):
            #     print('logit: ',logit[i],label[i])
            #assert 1==0

            labels.extend(label)
            preds.extend(logit)
            imp_indexes.extend(imp_index)
            print('all data: ',len(labels))

    group_labels, group_preds = group_labels_func(labels, preds, imp_indexes)
    res = cal_metric(group_labels, group_preds, metrics)
    return res['group_auc']

def train(model,optimizer, args):

    print('params: '," T_warm: ",T_warm," all_iteration: ",all_iteration," lr: ",lr)
    #writer = SummaryWriter('./model_snapshot_error')
    # cuda_list=range(cuda_num)
    cuda_list=range(args.size)
    #model.cuda(cudaid)
    # accumulation_steps=40
    accumulation_steps=int(args.batch_size/args.size/args.gpu_size)
    #accumulation_steps=1
    model = nn.DataParallel(model, device_ids=cuda_list)
    accum_batch_loss=0
    iterator=NewsIterator(batch_size=args.gpu_size*args.size, npratio=4,feature_file=os.path.join(args.data_dir,args.feature_file),field=args.field)
    train_file=os.path.join(args.data_dir, args.data_file)  
    #for epoch in range(0,100):
    batch_t=0
    iteration=0
    print('train...',cuda_list)
    #w=open(os.path.join(args.data_dir,args.log_file),'w')
    writer = SummaryWriter(os.path.join(args.data_dir, args.log_file) )
    
    epoch=0
    model.train()
    # batch_t=52880-1
    # iteration=3305-1
    best_score=-1
    batch_t=0
    iteration=0
    step=0
    # auc=test(model,args)
    # print(auc)
    #w=open(os.path.join(args.data_dir,args.log_file),'w')
    for epoch in range(10,20):
    #while True:
        all_loss=0
        all_batch=0
        data_batch=iterator.load_data_from_file(train_file)
        for  imp_index , user_index, his_id, candidate_id , label in data_batch:
            batch_t+=1
            assert candidate_id.shape[1]==2
            his_id=his_id.cuda(cudaid)
            candidate_id= candidate_id.cuda(cudaid)
            label = label.cuda(cudaid)
            loss,sample_size=model(his_id,candidate_id, label)

            sample_size=float(sample_size.sum())
            loss=loss.sum()/sample_size/math.log(2)
            # sample_size=float(sample_size)
            # loss=loss/sample_size/math.log(2)
            #print(' batch_t: ',batch_t, '  epoch: ',epoch,' loss: ',float(loss))
            print('???loss',loss)
            
            accum_batch_loss+=float(loss)

            all_loss+=float(loss)
            all_batch+=1

            loss = loss/accumulation_steps
            loss.backward()

            if (batch_t)%accumulation_steps==0:


                iteration+=1
                adjust_learning_rate(optimizer,iteration)
                
                optimizer.step()
                optimizer.zero_grad()
                print(' batch_t: ',batch_t, ' iteration: ', iteration, ' epoch: ',epoch,' accum_batch_loss: ',accum_batch_loss/accumulation_steps,' lr: ', optimizer.param_groups[0]['lr'])
                #w.write(' batch_t: '+str(batch_t)+' iteration: '+str(iteration)+' epoch: '+str(epoch)+' accum_batch_loss: '+str(accum_batch_loss/accumulation_steps)+'\n')
                writer.add_scalar('Loss/train', accum_batch_loss/accumulation_steps, iteration)
                writer.add_scalar('Ltr/train', optimizer.param_groups[0]['lr'], iteration)
                accum_batch_loss=0
                if iteration%500==0:
                    torch.cuda.empty_cache()
                    model.eval()
                    auc=test(model,args)
                    print(auc)
                    if auc>best_score:
                        #torch.save(model.state_dict(), os.path.join(args.save_dir,'Plain_robert_dot_best_0.pkl'))
                        best_score=auc
                        print('best score: ',best_score)
                        writer.add_scalar('auc/valid', auc, step)
                        step+=1
                    torch.cuda.empty_cache()
                    model.train()
        #torch.save(model.state_dict(), os.path.join(args.save_dir,'Plain_robert_dot'+str(epoch)+'.pkl'))
    #w.close()
            

if __name__ == '__main__':

    # cuda_num=int(sys.argv[1])
    random.seed(1)
    np.random.seed(1) 
    torch.manual_seed(1) 
    torch.cuda.manual_seed(1)
    #main()
    args = parse_args()
    mydict=utils.load_dict(os.path.join(args.data_dir,'roberta.base'))
    # model=Plain_bert(padding_idx=mydict['<pad>'],vocab_size=len(mydict))
    model=Plain_bert(args)

    #optimizer = torch.optim.Adam(model.parameters(), lr=lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0)
    optimizer = apex.optimizers.FusedLAMB(model.parameters(), lr=lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0,max_grad_norm=1.0)

    
    model_dict = model.state_dict()
    # for k, v in model_dict.items(): 
    #     print(k,v.size())
    # print('----------------------------------------------------')

    #roberta = RobertaModel.from_pretrained(os.path.join(args.data_dir,'roberta.base'), checkpoint_file='model.pt')
    #model_file=os.path.join(args.save_dir,args.model_file)
    # save_model=torch.load(model_file, map_location=lambda storage, loc: storage)

    # pretrained_dict={}
    # #for name,parameters in roberta.named_parameters():
    # for name in save_model:
    #     #print(name,':',save_model[name].size())
    #     # print(name,':',parameters.size())
    #     # if "layer_norm" in  name:
    #     #   print(name, save_model[name], save_model[name].shape)
    #     # if ( 'layers' in name ):
    #     #   pretrained_dict[name[31:]]=parameters
    #     # elif ('embed_positions.weight' in name or 'embed_tokens' in name or 'emb_layer_norm' in name):
    #     #   pretrained_dict[name[31:]]=parameters

    #     # if  'lm_head' not in name:
    #     #     pretrained_dict[name[31:]]=parameters
    #     pretrained_dict[name[7:]]=save_model[name]
    #     #pretrained_dict[name]=save_model[name]
    #     # elif 'lm_head.' in name:
    #     #     pretrained_dict[name[14:]]=parameters
    # # print('----------------------------------------------------------')
    # print(pretrained_dict.keys())
    # #assert 1==0

    # model_dict.update(pretrained_dict)
    # #print(model_dict.keys())
    # model.load_state_dict(model_dict)
    # for item in model.parameters():
    #   print(item.requires_grad)
        
    
    model.cuda(cudaid)
    train(model,optimizer,args)
    

            
























