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


class Plain_bert(nn.Module):#
    def __init__(self,args):
        super().__init__()
        embedding_dim=768
        self.roberta=RobertaForSequenceClassification.from_pretrained('roberta-base', return_dict=True,output_hidden_states=True)

        self.dense = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(768)
        self.init_weights(self.dense)



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

        batch_size,can_num,can_legth=candidate_id.shape
        batch_size,_,his_length=his_id.shape
        sample_size=candidate_id.shape[0]
        his_id=his_id.reshape(-1,his_id.shape[-1])
        candidate_id=candidate_id.reshape(-1,can_legth)

        # his_padding_mask = 1-his_id.eq(1).unsqueeze(-1).type_as(his_id)#
        # can_padding_mask=1-candidate_id.eq(1).unsqueeze(-1).type_as(candidate_id)

        his_padding_mask = 1-his_id.eq(1).type_as(his_id)#
        can_padding_mask=1-candidate_id.eq(1).type_as(candidate_id)

        outputs_his = self.roberta(input_ids=his_id, attention_mask=his_padding_mask, labels=None)
        #print('???',len(outputs_his.hidden_states))
        his_features=outputs_his.hidden_states[-1][:,0,:]

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
        sample_size=candidate_id.shape[0]
        his_id=his_id.reshape(-1,his_id.shape[-1])
        candidate_id=candidate_id.reshape(-1,can_legth)

        # his_padding_mask = 1-his_id.eq(1).unsqueeze(-1).type_as(his_id)#
        # can_padding_mask=1-candidate_id.eq(1).unsqueeze(-1).type_as(candidate_id)

        his_padding_mask = 1-his_id.eq(1).type_as(his_id)#
        can_padding_mask=1-candidate_id.eq(1).type_as(candidate_id)

        outputs_his = self.roberta(input_ids=his_id, attention_mask=his_padding_mask, labels=None)
        #print('???',len(outputs_his.hidden_states))
        his_features=outputs_his.hidden_states[-1][:,0,:]

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























