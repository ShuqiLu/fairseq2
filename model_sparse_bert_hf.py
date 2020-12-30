import torch
from torch import nn
# import torch_blocksparse as tbs
# from deepscale.pt.deepscale_constants import ROUTE_TRAIN
# import types
# from base.base_model import BaseModel
# from models.bert.masked_lm_loss import MaskedLMLoss
#from models.bert.modeling_bert import BertOnlyMLMHead, BertModel
from transformers.modeling_bert import BertModel, BertConfig, BertForMaskedLM, BertOnlyMLMHead  
# from models.rankbertv2.modeling_v2 import RankBERTV2Model
# from sparse_transformers import SparseBert, SparseBertForSequenceClassification
#from models.bert.modeling_bert import BertLayerNorm
from transformers.modeling_roberta import RobertaConfig, RobertaForSequenceClassification 
import torch.nn.functional as F

import torch
from torch import nn
import torch_blocksparse as tbs
import numpy as np
import types
#from models.bert.modeling_bert import BertOnlyMLMHead, BertModel
#from transformers.modeling_bert import BertModel, BertConfig, BertForMaskedLM, BertOnlyMLMHead  
from sparse_transformers import SparseBert, SparseRobertaForSequenceClassification



class Plain_bert(nn.Module):#
    def __init__(self,args):
        super().__init__()
        embedding_dim=768
        self.roberta=RobertaForSequenceClassification.from_pretrained('roberta-base', return_dict=True,output_hidden_states=True)
        
        #self.bert.config = sm.config
        #self.roberta.resize_token_embeddings(119567)
        #self.tie_weights()

        self.dense = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(768)
        self.init_weights(self.dense)
        field=args.field

        if field=='sparse_16_title':
            self.his_len=16
            self.set_len=32
        elif field=='sparse_60_title':
            self.his_len=60
            self.set_len=32
        elif field=='sparse_60_cat':
            self.his_len=60
            self.set_len=32
        elif field=='sparse_20_cat_abs':
            self.his_len=20
            self.set_len=96
        elif field=='sparse_120_title':
            self.his_len=120
            self.set_len=32
        elif field=='sparse_120_cat':
            self.his_len=120
            self.set_len=32
        elif field=='sparse_40_cat_abs':
            self.his_len=40
            self.set_len=96
        elif field=='sparse_60_cat_abs':
            self.his_len=60
            self.set_len=96
        elif field=='sparse_60_title_last':
            self.his_len=60
            self.set_len=32
        elif field=='sparse_60_cat_last':
            self.his_len=60
            self.set_len=32
        elif field=='sparse_80_title_reverse':
            self.his_len=80
            self.set_len=32
        elif field=='sparse_80_title_non_reverse':
            self.his_len=80
            self.set_len=32  
        elif field=='sparse_16_title_reverse':
            self.his_len=16
            self.set_len=32
        elif field=='sparse_16_title_non_reverse':
            self.his_len=16
            self.set_len=32  
        sm = SparseRobertaForSequenceClassification.from_pretrained('roberta-base', return_dict=True,output_hidden_states=True)
        
        if 'reverse' in field:
            sm.make_long_and_sparse(self.his_len*self.set_len, "variable", 16, False,[32]*int(self.set_len*self.his_len/512),[0])
            self.atten_mask=torch.zeros((self.his_len*self.set_len,self.his_len*self.set_len))
        elif 'last' in field:
            sm.make_long_and_sparse(self.his_len*self.set_len+10*64, "longformer", 16, True,self.set_len,list(range(0,int(self.set_len/16)*self.his_len,int(self.set_len/16))))
            self.atten_mask=torch.zeros((self.his_len*self.set_len+10*64,self.his_len*self.set_len+10*64))
        else:
            sm.make_long_and_sparse(self.his_len*self.set_len, "longformer", 16, True,self.set_len,list(range(0,int(self.set_len/16)*self.his_len,int(self.set_len/16))))
            self.atten_mask=torch.zeros((self.his_len*self.set_len,self.his_len*self.set_len))
        
        self.sparse_roberta = sm.roberta
        
        
        self.atten_mask[0,:]=1
        if 'non_reverse' in field:
            self.atten_mask[:,0]=1
            # self.atten_mask[0:512,0:512]=1
            # self.atten_mask[512:1024,512:1024]=1
            # self.atten_mask[1024:1536,1024:1536]=1
            # self.atten_mask[1536:2048,1536:2048]=1
            # self.atten_mask[2048:2560,2048:2560]=1
            for item in range(0,self.set_len*self.his_len,512):
                self.atten_mask[item:item+512,item:item+512]=1

        elif 'reverse' in field:
            self.atten_mask=None

        # if 'reverse' in field:
        #     self.atten_mask[:,0]=1
        #     for item in range(0,self.set_len*self.his_len,512):
        #         self.atten_mask[item:item+512,item:item+512]=1

        elif 'last' not in field:
            self.atten_mask[:,1]=1
            for item in range(0,int(self.set_len/16)*self.his_len,int(self.set_len/16)):
                start=item*16
                # self.atten_mask[start,:]=1#global
                self.atten_mask[:,start]=1
            for item in range(self.his_len):
                self.atten_mask[item*self.set_len:(item+1)*self.set_len,item*self.set_len:(item+1)*self.set_len]=1
        else:
            self.atten_mask[:,1]=1
            for item in range(0,int(self.set_len/16)*(self.his_len-10),int(self.set_len/16)):
                start=item*16
                self.atten_mask[:,start]=1

            for item in range(int(self.set_len/16)*(self.his_len-10),int(self.set_len/16)*(self.his_len-10)+int((self.set_len+64)/16)*10,int((self.set_len+64)/16)):
                start=item*16
                self.atten_mask[:,start]=1

            for item in range(self.his_len-10):
                self.atten_mask[item*self.set_len:(item+1)*self.set_len,item*self.set_len:(item+1)*self.set_len]=1
            for item in range(self.his_len-10,self.his_len):
                start=50*self.set_len+(item-50)*(self.set_len+64)
                end=start+(self.set_len+64)
                self.atten_mask[start:end,start:end]=1
                #print('????',start,end,self.atten_mask[start:end,start:end])

        self.field=field

        # torch.set_printoptions(profile="full")
        # w=open('atten_mask.txt','w')
        # w.write(str(self.atten_mask.cpu()))
        # w.close()
        

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        # elif isinstance(module, BertLayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        # if isinstance(module, nn.Linear) and module.bias is not None:
        #     module.bias.data.zero_()

    def forward(self,his_id , candidate_id,label,mode='train'):

        #print('???',self.atten_mask)
        
        # his_id=his_id[:,:,:1024]
        # print('his_id: ',his_id[:,:1024].shape)


        batch_size,can_num,can_legth=candidate_id.shape
        batch_size,_,his_length=his_id.shape

        if 'last' not in self.field:
            #print('???',his_length)
            assert his_length==self.his_len*self.set_len
        else:
            #print('???',his_length)
            assert his_length==self.his_len*self.set_len+10*64


        sample_size=candidate_id.shape[0]
        his_id=his_id.reshape(-1,his_id.shape[-1])
        candidate_id=candidate_id.reshape(-1,can_legth)

        his_padding_mask = 1-his_id.eq(1).type_as(his_id)#

        #print('???',his_padding_mask.shape,self.atten_mask.shape)
        if self.atten_mask!=None:
            #print('....mask....')
            his_padding_mask=his_padding_mask.unsqueeze(1)
            attn_mask=self.atten_mask.unsqueeze(0).repeat(batch_size,1,1)
            attn_mask=attn_mask.cuda()
            his_padding_mask=his_padding_mask*attn_mask

        # print('???',his_padding_mask.dim())

        # his_atten_mask[:,self.global_mask:]=0#除local size之外的是不能看见的
        # his_atten_mask[:,self.global_mask:]=0#block里面的token除第一个之外是不能看见其他的
        # his_atten_mask[self.global_mask,:]=0#和上一行对称的mask
        # his_atten_mask[self.local_mask,:]=0#对每一个token mask掉local size中超出本篇文档的部分
        # his_atten_mask[self.local_mask,:]=0#和上一条对称

        can_padding_mask=1-candidate_id.eq(1).type_as(candidate_id)

        #print('???',his_id.shape,his_padding_mask.shape)

        outputs_his = self.sparse_roberta(input_ids=his_id, attention_mask=his_padding_mask)
        # print('sparse_roberta: ', self.sparse_roberta.config.use_return_dict)
        # print('???',outputs_his,len(outputs_his))#,[x.shape for x in outputs_his])
        his_features=outputs_his.last_hidden_state[:,0,:]#[-1][:,0,:]

        outputs_can = self.roberta(input_ids=candidate_id, attention_mask=can_padding_mask, labels=None)
        can_features=outputs_can.hidden_states[-1][:,0,:]

        his_features=his_features.reshape(batch_size,1,his_features.shape[-1])
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
        #print('loss: ',loss)
        return loss
    def predict(self,his_id , candidate_id):
        batch_size,can_num,can_legth=candidate_id.shape
        batch_size,_,his_length=his_id.shape

        if 'last' not in self.field:
            #print('???',his_length)
            assert his_length==self.his_len*self.set_len
        else:
            #print('???',his_length)
            assert his_length==self.his_len*self.set_len+10*64


        sample_size=candidate_id.shape[0]
        his_id=his_id.reshape(-1,his_id.shape[-1])
        candidate_id=candidate_id.reshape(-1,can_legth)

        his_padding_mask = 1-his_id.eq(1).type_as(his_id)#

        #print('???',his_padding_mask.shape,self.atten_mask.shape)
        if self.atten_mask!=None:
            #print('....mask....')
            his_padding_mask=his_padding_mask.unsqueeze(1)
            attn_mask=self.atten_mask.unsqueeze(0).repeat(batch_size,1,1)
            attn_mask=attn_mask.cuda()
            his_padding_mask=his_padding_mask*attn_mask

        # print('???',his_padding_mask.dim())

        # his_atten_mask[:,self.global_mask:]=0#除local size之外的是不能看见的
        # his_atten_mask[:,self.global_mask:]=0#block里面的token除第一个之外是不能看见其他的
        # his_atten_mask[self.global_mask,:]=0#和上一行对称的mask
        # his_atten_mask[self.local_mask,:]=0#对每一个token mask掉local size中超出本篇文档的部分
        # his_atten_mask[self.local_mask,:]=0#和上一条对称

        can_padding_mask=1-candidate_id.eq(1).type_as(candidate_id)

        #print('???',his_id.shape,his_padding_mask.shape)

        outputs_his = self.sparse_roberta(input_ids=his_id, attention_mask=his_padding_mask)
        # print('sparse_roberta: ', self.sparse_roberta.config.use_return_dict)
        # print('???',outputs_his,len(outputs_his))#,[x.shape for x in outputs_his])
        his_features=outputs_his.last_hidden_state[:,0,:]#[-1][:,0,:]

        outputs_can = self.roberta(input_ids=candidate_id, attention_mask=can_padding_mask, labels=None)
        can_features=outputs_can.hidden_states[-1][:,0,:]

        his_features=his_features.reshape(batch_size,1,his_features.shape[-1])
        can_features=can_features.reshape(batch_size,can_num,can_features.shape[-1]) 


        his_features = self.dense(his_features)
        his_features = self.layer_norm(his_features)

        can_features = self.dense(can_features)
        can_features = self.layer_norm(can_features)


        res=torch.matmul(his_features,can_features.transpose(1,2))
        res=res.squeeze(1)
        return res























