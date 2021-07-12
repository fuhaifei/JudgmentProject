import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as Data

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from doc2vec_model import get_doc2vec, get_doc_vec
from preprocess import get_vocab, load_label_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 分类标签
LABELS = {1: "标题案号", 2: "当事人、辩护人、被害人情况", 3: "案件始末", 4: "指控", 5: "证实文件", 6: "辩护意见", 7: "事实",
          8: "证据列举", 9: "判决结果", 10: "尾部", 11: "法律条文等附录"}
LABEL_NUMBER = 11

# doc2vec训练模型超参，没用就是占位
VECTOR_SIZE = 50
MIN_COUNT = 2
EPOCHS = 500
SEG_FILE_PATH = "./files/seg_files"
MODEL_PATH = ".\\files\\model_file\\doc2vec_temp1.model"
# 基于线性softmax的分类算法，超参
BATCH_SIZE = 128
SOFTMAX_EPOCH = 300
LR = 0.04


def get_init_data():
    """
    doc2vec模型转化向量
    :return: 源数据，对应的向量表示
    """
    model = get_doc2vec(VECTOR_SIZE, MIN_COUNT, EPOCHS, MODEL_PATH, SEG_FILE_PATH, 1)
    docs, labels = load_label_data()
    doc_vec = get_doc_vec(docs, model)
    return docs, doc_vec, labels


def init_train_data(doc_vec, labels):
    """
    :param doc_vec: doc2vec抽取的文章向量
    :param labels: 对应的标签
    :return: 封装好的训练和测试集dataset
    """
    # 首先将数组展开为nums * features形式
    doc_vec_flatten = [sentence for doc in doc_vec for sentence in doc]
    doc_labels_flatten = [sentence for doc in labels for sentence in doc]
    # 82开划分训练集和测试集合
    train_vec, test_vec, train_label, test_label = train_test_split(doc_vec_flatten, doc_labels_flatten, test_size=0.2)
    train_set = MyDataset(torch.Tensor(train_vec), torch.Tensor(train_label))
    test_set = MyDataset(torch.Tensor(test_vec), torch.Tensor(test_label))
    return train_set, test_set


def get_conf_matrix(true_labels, predict_labels):
    """
    生成混淆矩阵和对应的图片
    :param true_labels:
    :param predict_labels:
    :return:
    """
    conf_mx = confusion_matrix(true_labels, predict_labels)
    conf_sum = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / conf_sum
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.show()
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.show()
    return conf_mx




"""
1.基于pytroch的线性softmax模型划分
"""


class SoftMaxClassificationModel(nn.Module):
    """
    softmax分类模型
    """

    def __init__(self, n_features, n_labels):
        super(SoftMaxClassificationModel, self).__init__()
        self.linear = nn.Linear(n_features, n_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.linear(x)


# 自定义数据读取类，可结合dataloader进行数据批量读取
class MyDataset(Data.Dataset):
    def __init__(self, features, labels):
        assert len(features) == len(labels)
        self.features = features
        self.labels = labels

    # 返回当前行数据
    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    # 返回数据长度
    def __len__(self):
        return len(self.features)


def train_net(net, train_iter, test_iter, loss_func, optimizer, num_epochs, device=None):
    """
    :param net: 待训练网络
    :param train_iter: 训练集
    :param test_iter: 测试集
    :param loss_func: 损失函数
    :param optimizer: 优化器，即蒜香传播计算机制
    :param num_epochs: 迭代epoch次数
    :param device: 计算设备
    :return:
    """
    if device is None:
        device = list(net.parameters())[0].device

    for i in range(num_epochs):
        totalNumber = 0
        totalLoss = 0
        time_start = time.time()

        for batch_features, batch_labels in train_iter:
            prediction = net(batch_features.to(device))
            loss = loss_func(prediction, batch_labels.to(device).long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            totalNumber += batch_features.shape[0]
            totalLoss += loss.item()
        time_end = time.time()
        print("epoch %d train loss is %f,train accuracy: %f,predict accuracy is %f,train time:%f" % (
            i + 1, totalLoss / totalNumber, compute_accuracy(train_iter, net, device),
            compute_accuracy(test_iter, net, device), time_end - time_start))


def compute_accuracy(test_iter, net, device=None):
    """
    :param test_iter: 测试集
    :param net: 训练完成的网络
    :param device:训练位置
    :return: 返回训练集准确率
    """
    if device is None:
        device = list(net.parameters())[0].device
    totalNumber = 0
    correctNumber = 0
    for batch_features, batch_labels in test_iter:
        prediction = net(batch_features.to(device))
        totalNumber += batch_features.shape[0]
        correctNumber += (prediction.argmax(dim=1) == torch.Tensor(batch_labels).to(device)).cpu().sum().item()
    return correctNumber / totalNumber


# docs, doc_vec, labels = get_init_data()
# train_set, test_set = init_train_data(doc_vec, labels)
# train_iter = Data.DataLoader(train_set, batch_size=BATCH_SIZE)
# test_iter = Data.DataLoader(test_set, batch_size=BATCH_SIZE)
# net = SoftMaxClassificationModel(50, 11)
# loss_func = nn.CrossEntropyLoss()
# optimizer  = torch.optim.SGD(net.parameters(),lr = LR)
# train_net(net, train_iter, test_iter, loss_func, optimizer, SOFTMAX_EPOCH, device=None)
"""
2. 基于机器学习的分类方法
"""


def get_train_test(test_size=0.25) -> object:
    docs, doc_vec, labels = get_init_data()
    doc_vec_flatten = [sentence for doc in doc_vec for sentence in doc]
    doc_labels_flatten = [sentence for doc in labels for sentence in doc]
    x_train, x_test, y_train, y_test = train_test_split(doc_vec_flatten, doc_labels_flatten, test_size=test_size)
    return x_train, x_test, y_train, y_test


def train_and_score(test_model, test_parameters):
    x_train, x_test, y_train, y_test = get_train_test()
    aim_Model = GridSearchCV(test_model, test_parameters)
    aim_Model.fit(x_train, y_train)
    print(aim_Model.best_score_)
    print(aim_Model.best_params_)
    print(aim_Model.score(x_test, y_test))
    return aim_Model.best_estimator_

# 1.线性svm分类
# linear_svm_parameters = {"model__C": [0.01, 0.1, 1, 2, 3], "model__max_iter": [1000000]}
# linear_svm_clf = Pipeline([
#     ("scaler", StandardScaler()),
#     ("model", LinearSVC(loss='hinge', ))
# ])

# 2.不同kernel的svm模型尝试
# kernel_svm_parameters = {"model__kernel": ["rbf", "linear", "poly"], "model__max_iter": [1000000]}
# kernel_svm_clf = Pipeline([
#     ("scaler", StandardScaler()),
#     ("model", SVC())
# ])

# 3.rbf内核测试
# kernel_svm_parameters = {"model__kernel": ['rbf'], "model__gamma": [0.01, 0.04 ,0.05, 0.06, 0.08],
#                          "model__C": [5, 6, 7, 8, 9], "model__max_iter": [1000000]}
# 4.多项式内核测试

# kernel_svm_parameters = {"model__kernel": ['poly'], "model__degree": [2, 3, 4], "model__C": [3, 4, 5, 6],
#                          "model__coef0": [0.1, 1, 2, 3], "model__max_iter": [1000000]}

"""

3. 基于决策树的分类算法的尝试
"""
# decision_tree_parameters = {'criterion': ["gini"], 'max_depth': [9, 10, 11, 12, 13],
#                             'min_samples_split': [3, 4, 5, 10], 'max_leaf_nodes': [40, 50, 60, 100, 200, 300]}


"""
4. K近邻分类算法
"""
# KNeighbors_clf_parameters = {"model__weights": ['distance'], 'model__n_neighbors': [4, 5, 6, 7, 8, 9, 10]}
# KNeighbors_svm_clf = Pipeline([
#     ("scaler", StandardScaler()),
#     ("model", KNeighborsClassifier())
# ])

"""
5. 随机森林 + 极端随机森林
"""
# rf_clf_parameters = {'n_estimators': [100, 200, 300],
#                      "n_jobs": [-1], 'oob_score': [True]}

# ef_clf_parameters = {'criterion': ['gini'],
#                      'max_depth': [20, 30, 40, 50, 60, 80, 100]}

"""
6. 集成学习的boosting方法（Adaboost 以及 GradientBoost两种方式
"""
# x_train, x_test, y_train, y_test = get_train_test()
# ada_clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=5),
#                              n_estimators=100, learning_rate=0.5)
# ada_clf.fit(x_train, y_train)
# print(ada_clf.score(x_train, y_train))
# print(ada_clf.score(x_test, y_test))
