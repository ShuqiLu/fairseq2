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
    ConcatSentencesDataset,
    OffsetTokensDataset,
    StripTokenDataset,
    RawLabelDataset


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



def get_batch(dataset,mydict,batch_size,decode_dataset=None,rerank=None,mode='train',dist=False,cudaid=0,size=1,start_pos=None):
   

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

    #data_size=len(dataset.sizes)
    print('size: ',data_size,len(src_dataset.sizes))
    assert len(dataset)==len(decode_dataset)
    assert len(rerank)==len(dataset)
    #assert 1==0

    valid_size=int(data_size*0.002)

    if mode=='train':
        data_size=data_size - valid_size
        rerank=rerank[:data_size]
    elif mode=='valid':
        data_size=valid_size
        rerank=rerank[-data_size:]
    else:
        assert 1==0

    if dist:
        assert size!=1
        dist_size=int(len(rerank)/size)+1
        rerank=rerank[cudaid*dist_size:(cudaid+1)*dist_size]
        print('dist_size: ',dist_size,cudaid)
        data_size=len(rerank)
        if start_pos!=None:
            i=start_pos

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

        # while i<data_size and length<batch_size*512: #index<batch_size:
        #     token_list.append( list(np.array(src_dataset.__getitem__(rerank[i]))))
        #     mask_label_list.append( list(np.array( tgt_dataset.__getitem__(rerank[i]))))
        #     decode_label_list.append(list(np.array( decode_dataset.__getitem__(rerank[i]))))
        #     #decode_label_list.append(  list(np.array(src_dataset.__getitem__(i))))

        #     if len(src_dataset.__getitem__(rerank[i]))>max_len_cur1:
        #         max_len_cur1=len(src_dataset.__getitem__(rerank[i]))
        #     if len(decode_dataset.__getitem__(rerank[i]))>max_len_cur2:
        #         max_len_cur2=len(decode_dataset.__getitem__(rerank[i]))

        #     length=max_len_cur1*index

        #     if max_len_cur1>512:
        #         max_len_cur1=512
        #     if max_len_cur2>512:
        #         max_len_cur2=512

        #     i+=1
        #     index+=1

        while i<data_size and length<=batch_size*512: #index<batch_size:

            old_max1=max_len_cur1
            old_max2=max_len_cur2
            if len(src_dataset.__getitem__(rerank[i]))>max_len_cur1:
                max_len_cur1=len(src_dataset.__getitem__(rerank[i]))
            if len(decode_dataset.__getitem__(rerank[i]))>max_len_cur2:
                max_len_cur2=len(decode_dataset.__getitem__(rerank[i]))
            if max_len_cur1>512:
                max_len_cur1=512
            if max_len_cur2>512:
                max_len_cur2=512

            index+=1

            if max_len_cur1>max_len_cur2:
                length=max_len_cur1*index
            else:
                length=max_len_cur2*index

            if length>batch_size*512:
                max_len_cur1=old_max1
                max_len_cur2=old_max2
                #print('???',batch_size,max_len_cur1,max_len_cur2,index,len(token_list))
                break
            # if cudaid==1:
            #     print('...',max_len_cur1,max_len_cur2,index,length)

            token_list.append( list(np.array(src_dataset.__getitem__(rerank[i]))))
            mask_label_list.append( list(np.array( tgt_dataset.__getitem__(rerank[i]))))
            decode_label_list.append(list(np.array( decode_dataset.__getitem__(rerank[i]))))

            i+=1


            
        #print(node_token_list[0],list(node_token_list[0]))
        #print([[ padding_node(node_token_list[0],max_node_len,mydict.pad()) ]])
        # node_token_list=torch.LongTensor([ padding_node(np.array(item),max_node_len,mydict['<pad>'])  for item in node_token_list ] )
        # node_mask_in_id=torch.LongTensor([ padding_node(np.array(item),max_node_len,mydict['<pad>'])  for item in node_mask_in_id ] )
        #print(token_list)
        # if cudaid==1:
        #     print('???',batch_size,max_len_cur1,max_len_cur2,index,len(token_list),' cudaid: ',cudaid)
        token_list=[padding(item,max_len=max_len_cur1, padding_idx=1) for item in token_list]
        mask_label_list=[padding(item,max_len=max_len_cur1, padding_idx=1) for item in mask_label_list]
        decode_label_list=[padding(item,max_len=max_len_cur2, padding_idx=1) for item in decode_label_list]



        token_list=torch.LongTensor(token_list)
        mask_label_list=torch.LongTensor(mask_label_list)
        decode_label_list=torch.LongTensor(decode_label_list)
        
        #print(node_token_list[0],node_mask_in_id[0])
        yield (token_list, mask_label_list,decode_label_list)
        #yield (token_list, mask_label_list)

def load_glue_data(task_path,mydict,mode='train'):#一个大列表，每个item是一个文档矩阵，矩阵里面每个item是一个node的数值  ，for token_id 和
    # dataset = data_utils.load_indexed_dataset(path,mydict,'mmap',combine=False,)
    # dataset = TokenBlockDataset(dataset,dataset.sizes,512 - 1,pad=mydict.pad(),eos=mydict.eos(), break_mode='complete',)
    # dataset = PrependTokenDataset(dataset, mydict.bos())
    #dataset=[]
    #input1=open(input_path1,'r').readlines()#[:10000]
    
    #label=open(label_path,'r').readlines()

    input0 = data_utils.load_indexed_dataset(
                os.path.join(task_path,'input0',mode),
                mydict,
                'mmap',
                combine=False,
            )
    assert input0 is not None, 'could not find dataset: {}'.format(get_path(type, split))

    input1 = data_utils.load_indexed_dataset(
                os.path.join(task_path,'input1',mode),
                mydict,
                'mmap',
                combine=False,
            )
    input0 = PrependTokenDataset(input0, mydict.bos())
    if input1 is None:
        src_tokens = input0
    else:
        input1 = PrependTokenDataset(input1, mydict.eos())
        src_tokens = ConcatSentencesDataset(input0, input1)


    if not 'STS-B' in task_path:
        label_dictionary=Dictionary.load(os.path.join(task_path,'label','dict.txt'))
        label_dictionary.add_symbol('<mask>')
        #label_dataset = make_dataset('label', label_dictionary)
        label_dataset= data_utils.load_indexed_dataset(
                os.path.join(task_path,'label',mode),
                label_dictionary,
                'mmap',
                combine=False,
            )
        if label_dataset is not None:
            
            label=OffsetTokensDataset(
                    StripTokenDataset(
                        label_dataset,
                        id_to_strip=label_dictionary.eos(),
                    ),
                    offset=-label_dictionary.nspecial,
                )
    else:
        label_path = "{0}.label".format( os.path.join(task_path,'label',mode))
        if os.path.exists(label_path):
            def parse_regression_target(i, line):
                values = line.split()
                assert len(values) == 1, \
                    f'expected num_classes={self.args.num_classes} regression target values on line {i}, found: "{line}"'
                return [float(x) for x in values]

            with open(label_path) as h:
                
                label=RawLabelDataset([
                        parse_regression_target(i, line.strip())
                        for i, line in enumerate(h.readlines())
                    ])
    print('data size: ',len(src_tokens),len(label))
    assert len(src_tokens)==len(label)
    # with data_utils.numpy_seed(self.args.seed):
    #     shuffle = np.random.permutation(len(src_tokens))

        # src_tokens = maybe_shorten_dataset(
        #     src_tokens,
        #     split,
        #     self.args.shorten_data_split_list,
        #     self.args.shorten_method,
        #     self.args.max_positions,
        #     self.args.seed,
        # )
    # input_data1=[]
    # input_data2=[]
    #label_list=[]
    # for line in input1:
    #     if len(line.strip())==0:
    #         input_data1.append([])
    #     else:
    #         line = line.strip().split(' ')
    #         input_data1.append([int(x) for x in line])
    # if input_path2:
    #     input2=open(input_path2,'r').readlines()
    #     for line in input2:
    #         if len(line.strip())==0:
    #             input_data2.append([])
    #         else:
    #             line = line.strip().split(' ')
    #             input_data2.append([int(x) for x in line])
    # if task=='QNLI':
    #     for line in label:
    #         line = line.strip()
    #         if line=='entailment':
    #             label_list.append(int(1))
    #         else:
    #             assert line=='not_entailment'
    #             label_list.append(int(0))
    # else:
    #     for line in label:
    #         line = line.strip()
    #         label_list.append(int(line))
    # print('data length: ',len(input_data1),len(input_data2))
    # assert len(input_data1)==len(label_list)
    # if len(input_data2)!=0:
    #     assert len(input_data1)==len(input_data2)


    return src_tokens,label

def get_batch_glue(dataset,label,mydict,batch_size,rerank,dist,cudaid,size,start_pos=None):
    # print('sort data...')
    # rerank_list,mask_list_all ,label_list_all=sort_data()
    # print('sort ok...')
    i=0
    # for item in src_dataset:
    #   print(item)
    #   break
    #print(src_dataset.__getitem__(0),src_dataset.__getitem__(0).shape)

    data_size=len(dataset)
    if rerank is None:
        rerank=np.arange(data_size)

    #data_size=len(dataset.sizes)
    print('size: ',data_size,len(label))
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
        label_list=[]
        #for  x in range(32):#感觉这里想拼起来不是那么容易，考虑一下
        index=0
        max_len_cur1=0
        max_len_cur2=0
        length=0

        # while i<data_size and length<=batch_size*512: #index<batch_size:
        while i<data_size:
            old_max1=max_len_cur1
            if len(dataset.__getitem__(rerank[i]))>max_len_cur1:
                max_len_cur1=len(dataset.__getitem__(rerank[i]))
            if max_len_cur1>512:
                max_len_cur1=512
            index+=1
            #length=max_len_cur1*index
            # if length>batch_size*512:
            #     max_len_cur1=old_max1
            #     break
            if index>batch_size:
                break
            token_list.append( list(np.array(dataset.__getitem__(rerank[i]))))
            label_list.append( list(np.array(label.__getitem__(rerank[i]))))
            #decode_label_list.append(  list(np.array(src_dataset.__getitem__(i))))

            i+=1
            
            
        #print(node_token_list[0],list(node_token_list[0]))
        #print([[ padding_node(node_token_list[0],max_node_len,mydict.pad()) ]])
        # node_token_list=torch.LongTensor([ padding_node(np.array(item),max_node_len,mydict['<pad>'])  for item in node_token_list ] )
        # node_mask_in_id=torch.LongTensor([ padding_node(np.array(item),max_node_len,mydict['<pad>'])  for item in node_mask_in_id ] )
        #print(token_list)

        token_list=[padding(item,max_len=max_len_cur1, padding_idx=1) for item in token_list]
        #label_list=[padding(item,max_len=max_len_cur1, padding_idx=1) for item in mask_label_list]

        token_list=torch.LongTensor(token_list)
        label_list=torch.LongTensor(label_list)
        
        #print(node_token_list[0],node_mask_in_id[0])
        yield (token_list, label_list)

















