import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, random_split
from transformers import BertModel, BertTokenizer, BertForSequenceClassification

from preprocess import load_label_data

# 哈工大中文bert模型,最高支持512长度句子
MODEL_NAME = "hfl/chinese-bert-wwm-ext"
MAX_SEQ_LENGTH = 150

# fine-tuning 的序列长度
BATCH_SIZE = 10
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-05


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


def prepare_train_data(sentences, tokenizer, max_seq_length=MAX_SEQ_LENGTH):
    """
    将输入句子转化为模型输入形式
    :param sentences: 输入段落
    :param tokenizer: 分词器
    :param max_seq_length: 最大输入序列长
    :return: 返回 序列， 序列mask， 序列type_ids
    """
    result_tokens = []
    result_masks = []
    result_token_ids = []
    for sentence in sentences:
        sentence = ''.join(sentence)
        # 对源句子进行切词和截断
        sentence_tokens = tokenizer.tokenize(sentence)
        if len(sentence_tokens) > max_seq_length - 2:
            sentence_tokens = sentence_tokens[:max_seq_length - 2]
        sentence_tokens = ['[CLS]'] + sentence_tokens + ['[SEP]']
        sentence_tokens = tokenizer.convert_tokens_to_ids(sentence_tokens)
        # 获取对应的mask和segment()
        sentence_padding = [0] * (max_seq_length - len(sentence_tokens))
        sentence_mask = [1] * len(sentence_tokens) + sentence_padding
        sentence_type_ids = [0] * len(sentence_tokens) + sentence_padding
        sentence_tokens += sentence_padding
        result_tokens.append(sentence_tokens)
        result_masks.append(sentence_mask)
        result_token_ids.append(sentence_type_ids)
    return result_tokens, result_masks, result_token_ids


def get_iter_dataset(sentence_tokens, sentence_labels, sentence_masks, sentence_type_ids, batch_size=BATCH_SIZE):
    """
    将预处理好的数据封装成可遍历的dataset_iter形式
    :param sentence_tokens:
    :param sentence_labels:
    :param sentence_masks:
    :param sentence_type_ids:
    :param batch_size:
    :return:
    """
    sentence_tokens_tensor = torch.LongTensor(sentence_tokens)
    sentence_labels_tensor = torch.LongTensor(sentence_labels)
    sentence_masks_tensor = torch.LongTensor(sentence_masks)
    sentence_type_ids_tensor = torch.LongTensor(sentence_type_ids)
    all_dataset = TensorDataset(sentence_tokens_tensor, sentence_labels_tensor, sentence_masks_tensor,
                                  sentence_type_ids_tensor)
    train_size = int(len(sentence_tokens) * 0.8)
    test_size = len(sentence_tokens) - train_size
    train_dataset, test_dataset = random_split(all_dataset, [train_size, test_size])
    train_sampler = RandomSampler(train_dataset)
    test_sampler = RandomSampler(test_dataset)
    train_iter = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
    return train_iter, test_iter


def fine_tuning_model(net, loss_func, optimizer, train_iter, test_iter, num_epochs=EPOCHS, device=DEVICE):
    """
    fine_tuning模型
    :param net: 预训练模型
    :param loss_func: 损失函数
    :param optimizer: 优化器
    :param train_iter: 训练集
    :param test_iter: 测试集
    :param num_epochs: 训练epoch次数
    :param device: 设备
    :return: null
    """
    # 确定训练设备
    if device is None:
        device = list(net.parameters())[0].device
    net.to(device)

    # fine_tuning_model
    for i in range(num_epochs):
        # 开启训练模式
        net.train()
        totalNumber = 0
        correctNumber = 0
        totalLoss = 0
        time_start = time.time()
        for sentence_tokens, sentence_labels, sentence_masks, sentence_type_ids in train_iter:
            prediction = net(sentence_tokens.to(device), sentence_masks.to(device), sentence_type_ids.to(device)).logits
            loss = loss_func(prediction, sentence_labels.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # 计算相关数据
            totalNumber += sentence_tokens.shape[0]
            totalLoss += loss.item()
            correctNumber += (prediction.argmax(dim=1) == sentence_labels.to(device)).cpu().sum().item()
        time_end = time.time()
        print("epoch %d train loss is %f,train accuracy: %f,predict accuracy is %f,train time:%f" % (
            i + 1, totalLoss / totalNumber, correctNumber / totalNumber,
            eval_net(net, test_iter), time_end - time_start))


def eval_net(net, test_iter, device=DEVICE):
    """
    使用测试集评价模型
    :param net: fine_tuning完成的模型
    :param test_iter: 测试集合
    :param device: 运行设备
    :return: 准确率
    """
    if device is None:
        device = list(net.parameters())[0].device
    net.to(device)
    net.eval()
    totalNumber = 0
    correctNumber = 0
    for sentence_tokens, sentence_labels, sentence_masks, sentence_type_ids in test_iter:
        prediction = net(sentence_tokens.to(device), sentence_masks.to(device), sentence_type_ids.to(device)).logits
        totalNumber += sentence_tokens.shape[0]
        correctNumber += (prediction.argmax(dim=1) == sentence_labels.to(device)).cpu().sum().item()
    return correctNumber / totalNumber


# # 初始化数据
# docs, labels = load_label_data()
# doc_flatten = [sentence for doc in docs for sentence in doc]
# doc_labels_flatten = [sentence for doc in labels for sentence in doc]
# tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
# result_tokens, result_masks, result_token_ids = prepare_train_data(doc_flatten, tokenizer, max_seq_length=MAX_SEQ_LENGTH)
# train_iter, test_iter = get_iter_dataset(result_tokens, doc_labels_flatten, result_masks, result_token_ids, batch_size=BATCH_SIZE)
#
# # 初始化模型
# net = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=11)
# loss_func = nn.CrossEntropyLoss()
# param_optimizer = list(net.named_parameters())
# no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
# # 避免对bias 和 layerNorm层正则化
# optimizer_grouped_parameters = [
#     {
#         'params':
#         [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
#         'weight_decay':
#         0.01
#     },
#     {
#         'params':
#         [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
#         'weight_decay':
#         0.0
#     }
# ]
# optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=LR)
#
# fine_tuning_model(net, loss_func, optimizer, train_iter, test_iter, num_epochs=EPOCHS)
# eval_net(net, test_iter, device=DEVICE)