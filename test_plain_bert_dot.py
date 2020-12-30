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
from model_plain_bert_dot3 import  Plain_bert
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
random.seed(1)
np.random.seed(1) 
torch.manual_seed(1) 
torch.cuda.manual_seed(1)


#cudaid=0
metrics=['group_auc','mean_mrr','ndcg@5;10']

def adjust_learning_rate(optimizer,iteration,lr=0.0005, T_warm=13345):#得看一些一共有多少个iteration再确定
    if iteration<=T_warm:
        lr=lr*float(iteration)/T_warm
    elif iteration<222414:
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = lr * (1 - (iteration - T_warm) / (222414 - T_warm))
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

    #w=open('temp_result.txt','w')
    # w2=open('temp_label.txt','w')
    # for k in sorted(all_keys):
    #     # mydict={i:group_preds[k][i] for i in range(len(group_preds[k]))}
    #     # mydict_sort=sorted(mydict.items(),key=lambda x : x[1],reverse=True)
    #     # mydict_sort2={mydict_sort[i][0]:i for i in range(len(mydict_sort)) }
    #     # mydict_sort2=[mydict_sort2[i]+1 for i in range(len(mydict_sort2))]
    #     # w.write(str(k)+' '+'['+','.join(str(item) for item in mydict_sort2)+']'+'\n')
    #     w2.write(str(k)+' '+'['+','.join(str(item) for item in group_labels[k])+']'+'\n')

    # w2.close()
    # #w.close()
    # print('write ok...')

    all_labels = []
    all_preds = []
    for k in all_keys:
        all_labels.append(group_labels[k])
        all_preds.append(group_preds[k])

    return all_labels, all_preds



def parse_args():
    parser = argparse.ArgumentParser("Transformer-XH")
    # parser.add_argument("--config-file", "--cf",
    #                 help="pointer to the configuration file of the experiment", type=str, required=True)

    parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                    "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--no_cuda",
                    default=False,
                    action='store_true',
                    help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                    default=False,
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
    parser.add_argument('--checkpoint',
                    type=int,
                    default=2500)
    parser.add_argument('--loss_scale',
                    type=float, default=0,
                    help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--test',
                    default=False,
                    action='store_true',
                    help="Whether on test mode")

    # parser.add_argument("--test-file", "--tf",
    #                 help="pointer to the configuration file of the experiment", type=str, required=True)
    parser.add_argument("--cudaid", "--cid",
                    help="pointer to the configuration file of the experiment", type=int, required=True)
    parser.add_argument("--data_dir",
                    type=str,
                    help="local_rank for distributed training on gpus")
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
    # parser.add_argument("--size",
    #                 type=int,
    #                 default=1,
    #                 help="local_rank for distributed training on gpus")
    # parser.add_argument("--batch_size",
    #                 type=int,
    #                 default=1,
    #                 help="local_rank for distributed training on gpus")
    parser.add_argument("--log_file",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--field",
                    type=str,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--can_length",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--gpu_size",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")


    return parser.parse_args()            

def test(model,args):#valid
    #w=open('train_plain_bert.txt','w')
    #model=Plain_bert(load_model='roberta.base',output_size=768)

    #model.cuda(cudaid)
    model.eval()
    test_file=os.path.join(args.data_dir, args.data_file)  
    cudaid=args.cudaid
    w=open(os.path.join(args.data_dir,args.log_file),'w')
    #data_batch=utils.get_batch()
    #cuda_list=range(cuda_num)
    #model = nn.DataParallel(model, device_ids=cuda_list)
    preds = []
    labels = []
    imp_indexes = []
    #test_file='valid_ms_roberta_plain.txt'
    #test_file='valid_ms_roberta_plain_large.txt'
    #test_file='valid_ms_roberta_plain.txt'
    feature_file=os.path.join(args.data_dir,args.feature_file)
    iterator=NewsIterator(batch_size=args.gpu_size, npratio=-1,feature_file=feature_file,field=args.field)
    print('test...')
    with torch.no_grad():
        data_batch=iterator.load_test_data_from_file(test_file,args.can_length)
        #data_batch=iterator.load_data_from_file(test_file)
        batch_t=0
        for  imp_index , user_index, his_id, candidate_id , label,can_len  in data_batch:
            batch_t+=len(candidate_id)
            # if batch_t<=167:
            #   continue
            his_id=his_id.cuda(cudaid)
            candidate_id= candidate_id.cuda(cudaid)
            #rank_mask=rank_mask.cuda(cudaid)
            #print(his_id.shape,candidate_id.shape,batch_t)
            # assert 1==0
            # imp_index= imp_index.cuda(cudaid)
            # label= label.cuda(cudaid)
            #print('???',his_id.shape,candidate_id.shape)
            #print('???',his_id)
            #print('???',candidate_id)
            logit=model.predict(his_id,candidate_id)
            # print('rank_mask: ',rank_mask)
            # print('user_index: ',user_index)
            # print('candidate_id: ',candidate_id)
            # print('batch_t: ',batch_t)
            #print('???',logit)
            #assert 1==0
            logit=np.array(logit.cpu())
            imp_index=np.reshape(np.array(imp_index), -1)
            assert len(imp_index)==len(logit)

            # logit=np.reshape(np.array(logit.cpu()), -1)
            # label=np.reshape(np.array(label), -1)
            # imp_index=np.reshape(np.array(imp_index), -1)
            #print('batch_t:',batch_t)
            for i in range(len(imp_index)):
                # w.write('imp_index:'+str(imp_index[i])+' '+' '.join([str(logit[i][j]) for j in range(can_len[i][0])]))
                # w.write('\n')
                for j in range(can_len[i][0]):
                    assert len(label[i])==can_len[i][0]
                    w.write('imp_index: '+str(imp_index[i])+' logit: '+str(logit[i][j])+' label: '+str(label[i][j])+'\n')
                    #print('imp_index: '+str(imp_index[i])+' logit: '+str(logit[i][j])+' label: '+str(label[i][j]))
                    
                # print('imp_index: '+str(imp_index[i])+' logit: '+str(logit[i])+' label: '+str(label[i]))
                # w.write('imp_index: '+str(imp_index[i])+' logit: '+str(logit[i])+' label: '+str(label[i])+'\n')
            #assert 1==0
            print('imp_index: ',imp_index[-1])
            # preds.extend(logit)
            # labels.extend(label)
            # imp_indexes.extend(imp_index)
            # batch_t+=len(candidate_id)
            #print(labels)
            #if batch_t==10:
            #break

        # group_labels, group_preds = group_labels_func(labels, preds, imp_indexes)
        # res = cal_metric(group_labels, group_preds, metrics)

    #return res
    w.close()

        # metrics:
     #  - group_auc
     #  - mean_mrr
     #  - ndcg@5;10

# def splice_file():
#   test_file='valid_ms_roberta_plain_large.txt'
#   f=open(test_file,'r').readlines()
#   #f=shuffle_data(test_file)
#   w1=open('valid_ms_roberta_plain_large0.txt','w')
#   w2=open('valid_ms_roberta_plain_large1.txt','w')
#   w3=open('valid_ms_roberta_plain_large2.txt','w')
#   w4=open('valid_ms_roberta_plain_large3.txt','w')
#   for line in f[:int(len(f)/4)]:
#       w1.write(line)

#   for line in f[int(len(f)/4):int(2*len(f)/4)]:
#       w2.write(line)

#   for line in f[int(2*len(f)/4):int(3*len(f)/4)]:
#       w3.write(line)

#   for line in f[int(3*len(f)/4):]:
#       w4.write(line)

#   w1.close()
#   w2.close()
#   w3.close()
#   w4.close()



def exact_result3():
    preds = []
    labels = []
    imp_indexes = []
    preds_t = []
    labels_t = []
    imp_indexes_t = []
    x=1
    flag=''
    count=0
    for num in [30,90,150,300]:
        #f1=open('/home/dihe/cudnn_file/recommender_shuqi/MIND_data/res_roberta_dot_abstract_6'+str(num)+'.txt','r').readlines() 
        # f1=open('../data/res_roberta_dot25'+str(num)+'.txt','r').readlines() #res_roberta_dot_abstract_63.txt
        f1=open('../data/res_dot3_fp16_'+str(num)+'.txt','r').readlines()
        for line in f1:
            line=line.strip().split(' ')
            logit=float(line[3])
            imp_index=int(line[1])
            label=int(float(line[5]))
            labels.append(label)
            preds.append(logit)
            imp_indexes.append(imp_index)
            if imp_index != flag:
                flag=imp_index
                count+=1
        #x=imp_indexes[-1]+1
    print('x: ',count,len(labels))
    group_labels, group_preds = group_labels_func(labels, preds, imp_indexes)

    res = cal_metric(group_labels, group_preds, metrics)
    print(res)




def exact_result(flag):
    preds = []
    labels = []
    imp_indexes = []
    f1=open('../data/res_roberta_dot'+str(flag)+'0.txt','r').readlines()
    i=0
    x=1
    for line in f1:

        if line[:9]=='imp_index':
            if '/home/shuqilu/' in line:
                # print('???',line,f1[i+1],f1[i+2],f1[i+3])

                # assert 1==0
                # logit=0.7214859
                # imp_index=94113+x
                # label=0
                logit=0.9787183
                imp_index=94112+x
                label=0
                preds.append(logit)
                labels.append(label)
                imp_indexes.append(imp_index)

            else:
                line=line.strip().split(' ')
                #print('???',line)
                logit=float(line[3])
                imp_index=int(line[1])+x
                label=int(float(line[5]))
                preds.append(logit)
                labels.append(label)
                imp_indexes.append(imp_index)
        i+=1

    f1=open('../data/res_roberta_dot'+str(flag)+'1.txt','r').readlines()
    i=0
    x=imp_indexes[-1]+1
    for line in f1:
        if line[:9]=='imp_index':
            if '/home/shuqilu/' in line:
                # print('???',line,f1[i+1],f1[i+2],f1[i+3])
                # assert 1==0
                # logit=0.9143183
                # imp_index=94117+x
                # label=0

                logit=0.9924859
                imp_index=94112+x
                label=0
                preds.append(logit)
                labels.append(label)
                imp_indexes.append(imp_index)
            else:
                line=line.strip().split(' ')
                logit=float(line[3])
                imp_index=int(line[1])+x
                label=int(float(line[5]))
                preds.append(logit)
                labels.append(label)
                imp_indexes.append(imp_index)
        i+=1

    f1=open('../data/res_roberta_dot'+str(flag)+'2.txt','r').readlines()
    i=0
    x=imp_indexes[-1]+1
    for line in f1:
        if line[:9]=='imp_index':
            if '/home/shuqilu/' in line:
                # print('???',line,f1[i+1],f1[i+2],f1[i+3])
                # assert 1==0
                # logit=0.9850529
                # imp_index=94116+x
                # label=1

                logit= 0.9541026
                imp_index=94116+x
                label=0
                preds.append(logit)
                labels.append(label)
                imp_indexes.append(imp_index)
            else:
                line=line.strip().split(' ')
                logit=float(line[3])
                imp_index=int(line[1])+x
                label=int(float(line[5]))
                preds.append(logit)
                labels.append(label)
                imp_indexes.append(imp_index)
        i+=1


    f1=open('../data/res_roberta_dot'+str(flag)+'3.txt','r').readlines()
    i=0
    x=imp_indexes[-1]+1
    for line in f1:
        if line[:9]=='imp_index':
            if '/home/shuqilu/' in line:
                # print('???',line,f1[i+1],f1[i+2],f1[i+3])
                # assert 1==0
                # logit=0.88046765
                # imp_index=94111+x
                # label=0

                logit=0.9569805
                imp_index=94117+x
                label=0
                preds.append(logit)
                labels.append(label)
                imp_indexes.append(imp_index)
            else:
                line=line.strip().split(' ')
                logit=float(line[3])
                imp_index=int(line[1])+x
                label=int(float(line[5]))
                preds.append(logit)
                labels.append(label)
                imp_indexes.append(imp_index)
        i+=1

    group_labels, group_preds = group_labels_func(labels, preds, imp_indexes)

    res = cal_metric(group_labels, group_preds, metrics)
    print(res)


if __name__ == '__main__':
    # splice_file()
    # flag=sys.argv[1]
    # exact_result(flag)
    
    # exact_result3()
    # assert 1==0



    #model_num=sys.argv[1]
    # cuda_num=int(sys.argv[1])
    args = parse_args()
    random.seed(1)
    np.random.seed(1) 
    torch.manual_seed(1) 
    torch.cuda.manual_seed(1)
    # cudaid=int(sys.argv[2])
    # test_file=sys.argv[3]
    #main()
    # mydict=utils.load_dict(os.path.join(args.data_dir,'roberta.base'))
    # model=Plain_bert(padding_idx=mydict['<pad>'],vocab_size=len(mydict))
    model=Plain_bert(args)

    iteration=0
    batch_t=0
    T_warm=10000
    epoch_o=0
    lr=0.0005

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0005,betas=(0.9,0.98),eps=1e-6,weight_decay=0.1)

    
    model_dict = model.state_dict()
    
    # for k, v in model_dict.items(): 
    #     print(k,v.size())
    # print('----------------------------------------------------')
    #print(model.state_dict()['score3.0.bias'])

    # roberta = RobertaModel.from_pretrained('./model/roberta.base', checkpoint_file='model.pt')
    model_file=os.path.join(args.save_dir,args.model_file)

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
        #pretrained_dict[name]=save_model[name]
        # if 'dense_lm.weight' in name[7:] or 'dense_lm.bias' in name[7:] :
        #     print(name ,save_model[name])
        # elif 'lm_head.' in name:
        #     pretrained_dict[name[14:]]=parameters
    # print('----------------------------------------------------------')
    print(pretrained_dict.keys())
    #assert 1==0
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda(args.cudaid)
    res=test(model,args)
    #print(res)
    #train(model,optimizer,cuda_num)
    # for epoch in range(5):
    #   iteration=train(model,epoch,optimizer,iteration,cuda_num)
    #   res=test(model)
    #   print(res)
        # optimizer.step()
        # optimizer.zero_grad()

            

























