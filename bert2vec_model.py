import torch
from preprocess import load_label_data
from transformers import BertModel, BertTokenizer

# 哈工大中文bert模型,最高支持512长度句子
MODEL_NAME = "hfl/chinese-bert-wwm-ext"


def get_cls_vec(docs):
    """
    使用bert模型 获得传入文章每一段的CLS的向量表示
    :param docs: 三位数组 [文章，段落，段落内容]
    :return: 三维数组 [文章，段落，cls向量]
    """
    result = []
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME)
    for doc in docs:
        doc_vec = []
        for sentence in doc:
            sentence = ''.join(sentence)
            # 由于模型输入最长为512，还需要添加cls和sep两个字符,所以截断为510
            if len(sentence) > 510:
                sentence = sentence[:510]
            sentence_token = tokenizer.encode(sentence)
            cls_vec = (model(torch.LongTensor([sentence_token]))[0][0][1]).detach().numpy().tolist()
            doc_vec.append(cls_vec)
        result.append(doc_vec)
    return result

