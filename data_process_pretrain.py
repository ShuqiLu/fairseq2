from datetime import datetime
# import bert
# from bert import run_classifier
# from bert import optimization
# from bert import tokenization
import csv
import numpy as np
import torch
import random
#from pytorch_transformers.tokenization_bert import BertTokenizer
import json
import os
import pickle
import sys
# from pytorch_transformers.modeling_bert import BertModel
from urllib.parse import urlparse



def get_abs():

    

def get_body():
    f=open('./corpus.train.tok.tmp2','r').readlines()
    w=open('./abs.txt','w')
    #先把空行什么的都去掉
    for line in f:
        #line=line.strip().split(' ')
        if line=='\n':
            continue
        else:
            line=line.strip().split('.')
            abs_sen=line[0]
            body_sen=line[1]
            abs_sen_t=abs_sen.split(' ')
            body_sen_t=body_sen.split(' ')
            if float(len(body_sen_t))/len(abs_sen_t)>=5:
                w.write(abs_sen+'.\n')
            else:
                w.write('\n')

def count_data():
    pass
if __name__ == '__main__':
    get_body()