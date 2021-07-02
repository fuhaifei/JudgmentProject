import torch
import torch.nn as nn
import torch.utils.data as Data
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SoftMaxClassificationModel(nn.Module):
    """
    softmax分类模型
    """

    def __init__(self, n_features, n_labels):
        super(SoftMaxClassificationModel, self).__init__()
        self.linear = nn.Linear(n_features, n_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.linear(x))


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


# 文件读取，标签还未定义完成，定义完成后进行读取
def load_data():
    pass


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
            loss = loss_func(prediction, batch_labels.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            totalNumber += batch_features.shape[0]
            totalLoss += loss.item()
        time_end = time.time()
        print("epoch %d train loss is %f predict accuracy is %f,train time:%f" % (
            i + 1, totalLoss / totalNumber, compute_accuracy(test_iter, net, device), time_end - time_start))


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
        correctNumber += (prediction.argmax(dim=1) == batch_labels.to(device)).cpu().sum().item()
    return correctNumber / totalNumber

