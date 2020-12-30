import json
import pickle
import numpy as np
import random
# from fairseq.data import Dictionary
# from fairseq import (
#   checkpoint_utils, distributed_utils, metrics, options, progress_bar, tasks, utils
# )
import torch
import argparse
import os
# from fairseq.models.roberta import RobertaModel
import random

random.seed(1)
np.random.seed(1) 
torch.manual_seed(1) 
torch.cuda.manual_seed(1)
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    mean_squared_error,
    accuracy_score,
    f1_score,
)

#from reco_utils.recommender.deeprec.io.iterator import BaseIterator


def mrr_score(y_true, y_score):
    """Computing mrr score metric.
    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.
    
    Returns:
        numpy.ndarray: mrr scores.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.
    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.
    Returns:
        numpy.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def hit_score(y_true, y_score, k=10):
    """Computing hit score metric at k.
    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.
    Returns:
        numpy.ndarray: hit score.
    """
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    for idx in argsort:
        if idx in ground_truth:
            return 1
    return 0


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.
    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.
    Returns:
        numpy.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


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
    f=open(os.path.join(path, 'dict.txt'))
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

def cal_metric(labels, preds, metrics):
    """Calculate metrics,such as auc, logloss.
    
    FIXME: 
        refactor this with the reco metrics and make it explicit.
    """
    res = {}
    for metric in metrics:
        if metric == "auc":
            auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
            res["auc"] = round(auc, 4)
        elif metric == "rmse":
            rmse = mean_squared_error(np.asarray(labels), np.asarray(preds))
            res["rmse"] = np.sqrt(round(rmse, 4))
        elif metric == "logloss":
            # avoid logloss nan
            preds = [max(min(p, 1.0 - 10e-12), 10e-12) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res["logloss"] = round(logloss, 4)
        elif metric == "acc":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            acc = accuracy_score(np.asarray(labels), pred)
            res["acc"] = round(acc, 4)
        elif metric == "f1":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            f1 = f1_score(np.asarray(labels), pred)
            res["f1"] = round(f1, 4)
        elif metric == "mean_mrr":
            mean_mrr = np.mean(
                [
                    mrr_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["mean_mrr"] = round(mean_mrr, 4)
        elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            for k in ndcg_list:
                ndcg_temp = np.mean(
                    [
                        ndcg_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["ndcg@{0}".format(k)] = round(ndcg_temp, 4)
        elif metric.startswith("hit"):  # format like:  hit@2;4;6;8
            hit_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                hit_list = [int(token) for token in ks[1].split(";")]
            for k in hit_list:
                hit_temp = np.mean(
                    [
                        hit_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["hit@{0}".format(k)] = round(hit_temp, 4)
        elif metric == "group_auc":
            #print([(each_labels, each_preds) for each_labels, each_preds in zip(labels, preds) ])
            group_auc = np.mean(
                [
                    roc_auc_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["group_auc"] = round(group_auc, 4)
        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res



def read_features_roberta(filename):

    news_id={}
    max_length=20
    # f=open(data_path+'/news_token_features_roberta.txt','r')
    f=open(filename,'r')
    for line in f:
        line=line.strip().split('\t')
        features=[int(x) for x in line[1:]]
        features=[0]+features

        if len(features)>max_length:
            features=features[:max_length]
        else:
            features=features+[1]*(max_length - len(features))
        news_id[line[0]]=features
    return news_id


class NewsIterator(object):
    """Train data loader for the NRMS NPA LSTUR model.
    Those model require a special type of data format, where each instance contains a label, impresion id, user id,
    the candidate news articlesand user's clicked news article. Articles are represented by title words. 

    Iterator will not load the whole data into memory. Instead, it loads data into memory
    per mini-batch, so that large files can be used as input data.

    Attributes:
        col_spliter (str): column spliter in one line.
        ID_spliter (str): ID spliter in one line.
        batch_size (int): the samples num in one batch.
        doc_size (int): max word num in news title.
        his_size (int): max clicked news num in user click history.
    """

    def __init__(self, batch_size, npratio, feature_file, col_spliter=" ", ID_spliter="%",mode='train'):
        """Initialize an iterator. Create necessary placeholders for the model.
        
        Args:
            hparams (obj): Global hyper-parameters. Some key setttings such as head_num and head_dim are there.
            graph (obj): the running graph. All created placeholder will be added to this graph.
            col_spliter (str): column spliter in one line.
            ID_spliter (str): ID spliter in one line.
        """
        self.col_spliter = col_spliter
        self.ID_spliter = ID_spliter
        self.batch_size = batch_size
        # self.doc_size = hparams.doc_size
        # self.his_size = hparams.his_size
        self.npratio = npratio
        # if mode=='train':
        #     self.news_dict=read_features_roberta('/home/shuqilu/Recommenders/data/data2/MINDlarge_train')
        # else:
        #     self.news_dict=read_features_roberta('/home/shuqilu/Recommenders/data/data2/MINDlarge_dev')
        #self.news_dict=read_features_roberta(feature_file)
        self.emb=np.load(feature_file)


    def parser_one_line(self, line):
        """Parse one string line into feature values.
        
        Args:
            line (str): a string indicating one instance

        Returns:
            list: Parsed results including label, impression id , user id, 
            candidate_news_index, clicked_news_index.
        """
        words = line.strip().split(self.ID_spliter)

        cols = words[0].strip().split(self.col_spliter)
        #label = [float(i) for i in cols[: self.npratio + 1]]
        # if self.npratio==-1:
        #     #label=[float(i) for i in cols[: self.npratio + 1]]
        #     pass
        # else:
        label=[0]
        candidate_news_index = []
        click_news_index = []
        imp_index = []
        user_index = []
        c_input_mask=[]
        c_segement=[]
        h_input_mask=[]
        h_segement=[]
        h_len=[]
        c_len=[]
        all_his=[]
        all_can=[]
        data_size=[]

        for news in cols:
            tokens = news.split(":")
            if "Impression" in tokens[0]:
                imp_index.append(int(tokens[1]))
            elif "User" in tokens[0]:
                #user_index.append(int(tokens[1]))
                w_temp=self.emb[int(tokens[1])]
                click_news_index.append(w_temp)

            elif "CandidateNewsPos" in tokens[0]:
                # word index start by 0
                #w_temp=[int(i) for i in tokens[1].split(",")]
                w_temp=self.emb[int(tokens[1])]
                candidate_news_index.append(w_temp)
                # count0=w_temp.count(0)
                # c_input_mask.append([1]*(len(w_temp)-count0)+[0]*count0)
                # c_segement.append([0]*len(w_temp))
            elif "CandidateNewsNeg" in tokens[0]:
                # word index start by 0
                # w_temp=[int(i) for i in tokens[1].split(",")]
                # candidate_news_index.append(w_temp)
                # count0=w_temp.count(0)
                # c_input_mask.append([1]*(len(w_temp)-count0)+[0]*count0)
                # c_segement.append([0]*len(w_temp))
                neg_list=tokens[1].split(",")
                neg_sample=np.random.choice(len(neg_list),1,replace=False)[0]
                w_temp=self.emb[int(neg_list[neg_sample])]
                candidate_news_index.append(w_temp)
                # w_temp=[int(i) for i in self.news_dict[neg_list[neg_sample]]]
                # candidate_news_index.append(w_temp)

            elif "CandidateNews" in tokens[0]:
                w_temp=self.emb[int(tokens[1])]
                candidate_news_index.append(w_temp)
                # word index start by 0
                # w_temp=[int(i) for i in tokens[1].split(",")]
                # candidate_news_index.append(w_temp)

            #     can_list=tokens[1].split(",")
            #     candidate_news_index=[self.news_dict[item] for item in can_list]
                # count0=w_temp.count(0)
                # c_input_mask.append([1]*(len(w_temp)-count0)+[0]*count0)
                # c_segement.append([0]*len(w_temp))

            # elif "ClickedNews" in tokens[0]:
            #     w_temp=[int(i) for i in tokens[1].split(",")]
            #     click_news_index.append(w_temp)
            #     #count0=w_temp.count(0)
            #     # h_input_mask.append([1]*(len(w_temp)-count0)+[0]*count0)
            #     # h_segement.append([0]*len(w_temp))
            # # elif "Hislen" in tokens[0]:
            #     # h_len=[[int(i)] for i in tokens[1].split(",")]
            #     # if len(h_len)!=50:
            #     #     print(len(h_len),h_len)
            #     # assert len(h_len)==50
            #     pass
                
            # elif "Canlen" in tokens[0]:
            #     #c_len=[[int(i)] for i in tokens[1].split(",")]
            #     #assert len(c_len)==5
            #     pass

            # elif "Allhis" in tokens[0]:
            #     all_his=[int(i) for i in tokens[1].split(",")]
            #     #assert all_his[0]!=0
            #     #.append(w_temp)
            #     #pass
            # elif "Allcan" in tokens[0]:
            #     all_can=[int(i) for i in tokens[1].split(",")]
            #     #.append(w_temp)

            elif "Label" in tokens[0]:
                label=[int(i) for i in tokens[1].split(",")]

            # elif "DataSize" in tokens[0]:
            #     data_size=[int(i) for i in tokens[1].split(",")]
                #.append(w_temp)
            else:
                raise ValueError("data format is wrong")
        # return (label, imp_index, user_index, candidate_news_index, click_news_index, c_input_mask,c_segement,h_input_mask,h_segement,h_len,c_len,all_his,all_can)
        if len(data_size)!=0:
            #print('its a valid data!!')
            return (label, imp_index, candidate_news_index, click_news_index)
        else:
            return (label, imp_index, candidate_news_index, click_news_index)

    def load_data_from_file(self, infile):
        """Read and parse data from a file.
        
        Args:
            infile (str): text input file. Each line in this file is an instance.

        Returns:
            obj: An iterator that will yields parsed results, in the format of graph feed_dict.
        """
        label_list = []
        imp_indexes = []
        user_indexes = []
        candidate_news_indexes = []
        click_news_indexes = []
        cnt = 0
        #input_ids=[]
        c_input_masks=[]
        c_segements=[]
        h_input_masks=[]
        h_segements=[]
        h_len=[]
        c_len=[]
        all_his=[]
        all_can=[]

        # with tf.gfile.GFile(infile, "r") as rd:
        with open(infile, "r") as rd:
            for line in rd:
                (
                    label,
                    imp_index,
                    candidate_news_index,
                    click_news_index,
                    # all_his_t,
                    # all_can_t,
                ) = self.parser_one_line(line)

                candidate_news_indexes.append(candidate_news_index)
                click_news_indexes.append(click_news_index)
                imp_indexes.append(imp_index)
                label_list.append(label)
                # c_input_masks.append(c_input_mask)
                # c_segements.append(c_segement)
                # h_input_masks.append(h_input_mask)
                # h_segements.append(h_segement)
                # h_len.append(h_len_t)
                # c_len.append(c_len_t)
                # all_his.append(all_his_t)
                # all_can.append(all_can_t)
                cnt += 1
                if cnt >= self.batch_size:
                    #input_mask=
                    yield self._convert_data(
                        label_list,
                        imp_indexes,
                        candidate_news_indexes,
                        click_news_indexes,
                        # c_input_masks,
                        # c_segements,
                        # h_input_masks,
                        # h_segements,
                        # h_len,
                        # c_len,
                        # all_his,
                        # all_can,
                    )
                    candidate_news_indexes = []
                    click_news_indexes = []
                    label_list = []
                    imp_indexes = []
                    user_indexes = []
                    #input_ids=[]
                    # c_input_masks=[]
                    # c_segements=[]
                    # h_input_masks=[]
                    # h_segements=[]
                    # h_len=[]
                    # c_len=[]
                    # all_his=[]
                    # all_can=[]
                    cnt = 0


    def load_test_data_from_file(self, infile):
        """Read and parse data from a file.
        
        Args:
            infile (str): text input file. Each line in this file is an instance.

        Returns:
            obj: An iterator that will yields parsed results, in the format of graph feed_dict.
        """
        label_list = []
        imp_indexes = []
        user_indexes = []
        candidate_news_indexes = []
        click_news_indexes = []
        cnt = 0
        c_input_masks=[]
        c_segements=[]
        h_input_masks=[]
        h_segements=[]
        h_len=[]
        c_len=[]
        all_his=[]
        all_can=[]
        data_index_record=''
        batch_size=1

        # with tf.gfile.GFile(infile, "r") as rd:
        with open(infile, "r") as rd:
            for line in rd:

                (
                    label,
                    imp_index,
                    user_index,
                    candidate_news_index,
                    click_news_index,
                    # all_his_t,
                    # all_can_t,
                    # data_size,
                ) = self.parser_one_line(line)
                #batch_size=data_size[0]
                candidate_news_indexes.append(candidate_news_index)
                #candidate_news_indexes=candidate_news_index
                candidate_news_indexes=np.array(candidate_news_indexes)
                #print('???',candidate_news_indexes)
                candidate_news_indexes=candidate_news_indexes.reshape(1,-1,len(candidate_news_index[0]))

                if len(click_news_indexes)==0:
                    click_news_indexes.append(click_news_index)
                    #all_his.append(all_his_t)

                # imp_indexes.append(imp_index)
                # user_indexes.append(user_index)
                # label_list.append(label)
                imp_indexes=[imp_index[0]]*len(candidate_news_index)
                user_indexes=[user_index[0]]*len(candidate_news_index)
                label_list=label

                # c_input_masks.append(c_input_mask)
                # c_segements.append(c_segement)
                # h_input_masks.append(h_input_mask)
                # h_segements.append(h_segement)
                # h_len.append(h_len_t)
                # c_len.append(c_len_t)
                
                #all_can.append(all_can_t)
                cnt += 1
                if cnt >= batch_size:
                    

                    yield self._convert_data(
                        label_list,
                        imp_indexes,
                        user_indexes,
                        candidate_news_indexes,
                        click_news_indexes,
                        # c_input_masks,
                        # c_segements,
                        # h_input_masks,
                        # h_segements,
                        # h_len,
                        # c_len,
                        # all_his,
                        # all_can,
                    )
                    candidate_news_indexes = []
                    click_news_indexes = []
                    label_list = []
                    imp_indexes = []
                    user_indexes = []
                    #input_ids=[]
                    # c_input_masks=[]
                    # c_segements=[]
                    # h_input_masks=[]
                    # h_segements=[]
                    # h_len=[]
                    # c_len=[]
                    all_his=[]
                    all_can=[]
                    cnt = 0

    def _convert_data(
        self,
        label_list,
        imp_indexes,
        candidate_news_indexes,
        click_news_indexes,


    ):
        """Convert data into numpy arrays that are good for further model operation.
        
        Args:
            label_list (list): a list of ground-truth labels.
            imp_indexes (list): a list of impression indexes.
            user_indexes (list): a list of user indexes.
            candidate_news_indexes (list): the candidate news article's words indices
            click_news_indexes (list): words indices for user's clicked news articles
            
        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """

        # labels = np.asarray(label_list, dtype=np.float32)
        # imp_indexes = np.asarray(imp_indexes, dtype=np.int32)
        # user_indexes = np.asarray(user_indexes, dtype=np.int32)
        # candidate_news_index_batch = np.asarray(candidate_news_indexes, dtype=np.int32)
        # click_news_index_batch = np.asarray(click_news_indexes, dtype=np.int32)
        labels = torch.LongTensor(label_list)
        imp_indexes = torch.LongTensor(imp_indexes) 
        candidate_news_index_batch = torch.FloatTensor(candidate_news_indexes) 
        click_news_index_batch = torch.FloatTensor(click_news_indexes) 
        # c_input_masks=np.asarray(c_input_masks, dtype=np.int32)
        # c_segements=np.asarray(c_segements, dtype=np.int32)
        # h_input_masks=np.asarray(h_input_masks, dtype=np.int32)
        # h_segements=np.asarray(h_segements, dtype=np.int32)
        # c_length=np.asarray(c_length,dtype=np.int32)
        # h_length=np.asarray(h_length,dtype=np.int32)
        # all_his=np.asarray(all_his,dtype=np.int32)
        # all_can=np.asarray(all_can,dtype=np.int32)
        # all_his = torch.LongTensor(all_his)  
        # all_can = torch.LongTensor(all_can)  
        # return {
        #     "impression_index_batch": imp_indexes,
        #     "user_index_batch": user_indexes,
        #     "clicked_news_batch": click_news_index_batch,
        #     "candidate_news_batch": candidate_news_index_batch,
        #     # "c_input_masks":c_input_masks,
        #     # "c_segments":c_segements,
        #     # "h_input_masks":h_input_masks,
        #     # "h_segments":h_segements,
        #     "labels": labels,
        #     "c_length":c_length,
        #     "h_length":h_length,
        #     "all_his":all_his,
        #     "all_can":all_can,
        # }
        return (
            imp_indexes,
            click_news_index_batch,
            candidate_news_index_batch,
            # "c_input_masks":c_input_masks,
            # "c_segments":c_segements,
            # "h_input_masks":h_input_masks,
            # "h_segments":h_segements,
            labels,
            # all_his,
            # all_can,
        )
