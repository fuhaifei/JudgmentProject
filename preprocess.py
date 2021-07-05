import collections

from pylab import *
import os

# 两个分词模型
import jieba
import pkuseg
import torchtext.vocab as Vocab
import torch
import pandas as pd
import numpy as np
from pandas.plotting import table

mpl.rcParams['font.sans-serif'] = ['SimHei']

"""
全局属性
"""
# 判决书路径
FILE_PATH = 'E:\项目相关资料\法律文书项目\Judgement\Data.case.txt'
# 分词文件存储路径
SEG_FILE_PATH = "./files/seg_files"
# 标准文件地址
LABEL_FILE_PATH = "./files/label_files"
# 构建词典的最小词频
MIN_FREQ = 5
# 段落标签
LABELS = {1: "标题案号", 2: "当事人、辩护人、被害人情况", 3: "案件始末", 4: "指控", 5: "证实文件", 6: "辩护意见", 7: "事实",
          8: "证据列举", 9: "判决结果", 10: "尾部", 11: "法律条文等附录"}


def word_segment(aim_sentence, options=1, stop_word="./files/stop_word_file/cn_stopwords.txt"):
    """
    分词模型函数
    :param stop_word:
    :param aim_sentence:
    :param options: 模型选择参数
    :return: 返回分词+去掉停用词结果
    """
    seg_result = []
    result = []
    if options == 1:
        seg_result = jieba.lcut(aim_sentence, cut_all=False)
    elif options == 2:
        seg = pkuseg.pkuseg()
        seg_result = seg.cut(aim_sentence)
    # 去除停用词
    stopwords = [line.strip() for line in open(stop_word, encoding="utf-8").readlines()]
    for word in seg_result:
        if word not in stopwords:
            result.append(word)
    return result


def read_judgements(file_path, out_path=SEG_FILE_PATH):
    """
    文件读取+分词
    :param out_path: 分词结果存储文件
    :param file_path: 文件路径
    """
    for dir_name in os.listdir(file_path):
        if os.path.isdir(os.path.join(file_path, dir_name)):
            # 遍历文件读取句子
            for file_name in os.listdir(os.path.join(file_path, dir_name)):
                # 打开文件
                if os.path.splitext(os.path.join(file_path, dir_name, file_name))[1] == '.txt':
                    with open(os.path.join(file_path, dir_name, file_name), 'r', encoding='utf-8') as f:
                        file_lines = []
                        file_line = f.readline()
                        while file_line:
                            # 跳过尾部的垃圾数据行
                            if file_line.startswith('附件') or file_line.startswith('退赔清单') or file_line.startswith(
                                    '附表') or file_line.startswith('序号'):
                                break
                            # 去掉数据中的空行
                            file_line = file_line.replace('\n', '').replace('\r', '').replace(' ', '')
                            if len(file_line) <= 1:
                                file_line = f.readline()
                                continue
                            file_lines.append(word_segment(file_line))
                            file_line = f.readline()
                        # 写入到分词结果文件
                        with open(os.path.join(out_path, "seg_" + file_name),
                                  'w', encoding='utf-8') as seg_file:
                            for line in file_lines:
                                seg_file.write(" ".join(line) + "\n")


def get_vocab(data_root=SEG_FILE_PATH):
    """
    :param data_root: 分词文件地址
    :return: 原始数据+词汇表
    """
    tokenized_data = []
    # 遍历读取所有文件
    for file_name in os.listdir(data_root):
        file_data = []
        with open(os.path.join(data_root, file_name), 'r', encoding='utf-8') as seg_file:
            file_line = seg_file.readline()
            while file_line:
                file_data.append(file_line.split())
                file_line = seg_file.readline()
        tokenized_data.append(file_data)
    # 展开为word一维数组
    words = [word for rows in tokenized_data for row in rows for word in row]
    counter = collections.Counter(words)
    # 调用tochtext的Vocab函数，封装为Vocab对象，包含dic，index，char数组属性
    return tokenized_data, Vocab.Vocab(counter, min_freq=MIN_FREQ)


def load_label_data(label_file_path=LABEL_FILE_PATH, seg_file_path=SEG_FILE_PATH):
    docs = []
    labels = []
    # 根据标签文件读取对应的分词文件
    for file_name in os.listdir(LABEL_FILE_PATH):
        if os.path.splitext(file_name)[1] == '.txt':
            doc_sentences = []
            doc_labels = []
            with open(os.path.join(label_file_path, file_name), 'r', encoding='utf-8') as labels_file:
                with open(os.path.join(seg_file_path, "seg_"+file_name), 'r', encoding='utf-8') as doc_file:
                    sentence_label = labels_file.readline().replace("\n", "").replace("\r", "")
                    sentence = doc_file.readline()
                    while sentence_label is not None and len(sentence_label) != 0:
                        doc_labels.append(int(sentence_label) - 1)
                        doc_sentences.append(sentence.split())
                        sentence_label = labels_file.readline().replace("\n", "").replace("\r", "")
                        sentence = doc_file.readline()
            docs.append(doc_sentences)
            labels.append(doc_labels)
    return docs, labels






# tokenized_data, vocab = get_vocab()
