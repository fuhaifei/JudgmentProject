import torch
from transformers import BertModel, BertTokenizer

# 哈工大中文bert模型,最高支持512长度句子
MODEL_NAME = "hfl/chinese-bert-wwm-ext"

def get_cls_vec():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME)
    tokenizer.encode("今天是个好日子心想的事儿都能成")
