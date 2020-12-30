import json
import pickle
import numpy as np
import random
from fairseq.data import Dictionary
from fairseq import (
  checkpoint_utils, distributed_utils, metrics, options, progress_bar, tasks, utils
)
import torch
import argparse
import os
# from fairseq.models.roberta import RobertaModel
import random
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
random.seed(1)
np.random.seed(1) 
torch.manual_seed(1) 
torch.cuda.manual_seed(1)
import re
SPACE_NORMALIZER = re.compile(r"\s+")



def load_dict(path):

    #mydict = Dictionary.load(os.path.join(path, 'dict.txt'))
    mydict={}
    cnt=0
    mydict['<s>']=cnt
    cnt+=1
    mydict['<pad>']=cnt
    cnt+=1
    mydict['</s>']=cnt
    cnt+=1
    mydict['<unk>']=cnt
    cnt+=1
    #mask_idx = mydict.add_symbol('<mask>')
    f=open(os.path.join(path, 'roberta.base/dict.txt'))
    lines = f.readlines()
    #indices_start_line = self._load_meta(lines)
    for line in lines:
        idx = line.rfind(" ")
        if idx == -1:
            raise ValueError(
                "Incorrect dictionary format, expected '<token> <cnt>'"
            )
        word = line[:idx]
        count = int(line[idx + 1 :])
        mydict[word] = cnt
        # self.symbols.append(word)
        # self.count.append(count)
        cnt+=1
    mydict['<mask>']=cnt
    cnt+=1
    print('load dict ok...')
    return  mydict


def load_mask_data(path,mydict):#一个大列表，每个item是一个文档矩阵，矩阵里面每个item是一个node的数值  ，for token_id 和
    #print('???',path)
    #from fairseq.data.indexed_dataset import MMapIndexedDataset
    #print('???', MMapIndexedDataset(path) )
    dataset = data_utils.load_indexed_dataset(path,mydict,'mmap',combine=False,)
    #print(dataset.__getitem__(0),dataset.__getitem__(0).shape,len(dataset))
    dataset = TokenBlockDataset(dataset,dataset.sizes,512 - 1,pad=mydict.pad(),eos=mydict.eos(), break_mode='complete_doc',)
    #print(dataset.__getitem__(0),dataset.__getitem__(0).shape,len(dataset))
    dataset = PrependTokenDataset(dataset, mydict.bos())
    #print(dataset.__getitem__(0),dataset.__getitem__(0).shape,len(dataset))
    
    return dataset

#不知道有没有shuffle的操作

    
def load_decode_data(path,mydict):

    dataset = data_utils.load_indexed_dataset(path,mydict,'mmap',combine=False,)
    dataset = PrependTokenDataset(dataset, mydict.bos())
    return dataset

def padding(mylist,max_len=512,padding_idx=1):

    if len(mylist)==2:
        mylist=[0]+[padding_idx]*(max_len-1)
    elif len(mylist)>max_len:
        # print('???',len(mylist),max_len)
        # assert 1==0
        mylist=mylist[:max_len]
    else:
        mylist+=[padding_idx]*(max_len-len(mylist))

    return mylist



def get_batch(dataset,mydict,batch_size,decode_dataset=None,rerank=None,dist=False,cudaid=0,size=1):
   

    src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(dataset,mydict,pad_idx=mydict.pad(),mask_idx=mydict.index('<mask>'),seed=1, mask_prob=0.15,leave_unmasked_prob=0.1,random_token_prob=0.1,freq_weighted_replacement=False,mask_whole_words=None,)
    i=0
    # for item in src_dataset:
    #   print(item)
    #   break
    #print(src_dataset.__getitem__(0),src_dataset.__getitem__(0).shape)


    src_dataset=PadDataset( src_dataset,pad_idx=mydict.pad(), left_pad=False,)
    #print(src_dataset.__getitem__(0),src_dataset.__getitem__(0).shape)
    tgt_dataset=PadDataset( tgt_dataset,pad_idx=mydict.pad(), left_pad=False,)
    #print('???',tgt_dataset.__getitem__(0))

    data_size=len(dataset)
    if rerank is None:
        rerank=np.arange(data_size)

    #data_size=len(dataset.sizes)
    print('size: ',data_size,len(src_dataset.sizes))
    #assert len(dataset)==len(decode_dataset)
    assert len(rerank)==len(dataset)
    #assert 1==0

    # valid_size=int(data_size*0.002)

    # if mode=='train':
    #     data_size=data_size - valid_size
    #     rerank=rerank[:data_size]
    # elif mode=='valid':
    #     data_size=valid_size
    #     rerank=rerank[-data_size:]
    # else:
    #     assert 1==0


    if dist:
        assert size!=1
        dist_size=int(len(rerank)/size)+1
        rerank=rerank[cudaid*dist_size:(cudaid+1)*dist_size]
        print('dist_size: ',dist_size,cudaid)
        data_size=len(rerank)

    index=0
    length=0

    while i < data_size:
        token_list=[]
        mask_label_list=[]
        decode_label_list=[]
        #for  x in range(32):#感觉这里想拼起来不是那么容易，考虑一下
        index=0
        max_len_cur1=0
        max_len_cur2=0
        length=0

        while i<data_size and length<=batch_size*512: #index<batch_size:
            old_max1=max_len_cur1
            if len(src_dataset.__getitem__(rerank[i]))>max_len_cur1:
                max_len_cur1=len(src_dataset.__getitem__(rerank[i]))
            if max_len_cur1>512:
                max_len_cur1=512
            index+=1
            length=max_len_cur1*index
            if length>batch_size*512:
                max_len_cur1=old_max1
                break
            token_list.append( list(np.array(src_dataset.__getitem__(rerank[i]))))
            mask_label_list.append( list(np.array( tgt_dataset.__getitem__(rerank[i]))))
            #decode_label_list.append(  list(np.array(src_dataset.__getitem__(i))))

            i+=1
            
            
        #print(node_token_list[0],list(node_token_list[0]))
        #print([[ padding_node(node_token_list[0],max_node_len,mydict.pad()) ]])
        # node_token_list=torch.LongTensor([ padding_node(np.array(item),max_node_len,mydict['<pad>'])  for item in node_token_list ] )
        # node_mask_in_id=torch.LongTensor([ padding_node(np.array(item),max_node_len,mydict['<pad>'])  for item in node_mask_in_id ] )
        #print(token_list)

        token_list=[padding(item,max_len=max_len_cur1, padding_idx=1) for item in token_list]
        mask_label_list=[padding(item,max_len=max_len_cur1, padding_idx=1) for item in mask_label_list]



        token_list=torch.LongTensor(token_list)
        mask_label_list=torch.LongTensor(mask_label_list)
        
        #print(node_token_list[0],node_mask_in_id[0])
        yield (token_list, mask_label_list)
        #yield (token_list, mask_label_list)

















