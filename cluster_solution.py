import os.path

from preprocess import get_vocab
from doc2vec_model import get_doc2vec
from doc2vec_model import get_doc_vec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
# 两行解决中文乱码问题
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

IMAGE_PATH = "./files/out_image"
SEG_FILE_PATH = "./files/seg_files"
MODEL_PATH = ".\\files\\model_file\\doc2vec_temp1.model"
# 训练模型超参
VECTOR_SIZE = 50
MIN_COUNT = 2
EPOCHS = 100


def get_doc2vec_init_data():
    """
    doc2vec模型转化向量
    :return: 源数据，对应的向量表示
    """
    model = get_doc2vec(VECTOR_SIZE, MIN_COUNT, EPOCHS, MODEL_PATH, SEG_FILE_PATH, 1)
    tokenized_data, vocab = get_vocab()
    doc_vec = get_doc_vec(tokenized_data, model)
    return tokenized_data, doc_vec


def cluster_doc(cluster_data, kernels):
    all_models = []
    scores = []
    silhouette_scores = []
    for kernel in kernels:
        cluster_model = KMeans(n_clusters=kernel, init='random')
        cluster_model.fit(cluster_data)
        all_models.append(cluster_model)
        scores.append(cluster_model.score(cluster_data, cluster_model.labels_))
        silhouette_scores.append(silhouette_score(cluster_data, cluster_model.labels_))
    # 绘制函数图片
    fig = plt.figure()
    sub_fig1 = fig.add_subplot(211)
    sub_fig1.plot(kernels, scores, linestyle="--", marker="o", linewidth=1.0)
    sub_fig1.set_title('平均距离')
    sub_fig1.set_ylabel('距离大小')
    sub_fig2 = fig.add_subplot(212)
    sub_fig2.plot(kernels, silhouette_scores, linestyle="--", marker="o", linewidth=1.0)
    sub_fig2.set_xlabel("kernels")
    sub_fig2.set_ylabel('轮廓系数大小')
    plt.savefig(os.path.join(IMAGE_PATH, 'doc2vec_cluster.png'), dpi=300)
    plt.show()
    return all_models


if __name__ == "__main__":
    tokenized_data, doc_vec = get_doc2vec_init_data()
    # 将文章形式组织的段落展开为段落
    train_vec = [sentence for doc in doc_vec for sentence in doc]
    print(len(train_vec))
    print("训练聚类模型")
    all_models = cluster_doc(train_vec, [6, 7, 8, 9, 10, 11])
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter = 1000)