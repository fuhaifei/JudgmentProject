# JudgementProject

法律文书项目相关，主要实验方向有两个

1. 通过doc2vec抽取特征向量
2. 通过bert抽取特征向量

完成之后，通过分类模型和聚类模型查看方案可行性


## 当前阶段完成任务
* 数据预处理(preprocess.py)
  1. 结巴（jieba）分词,对原文件分词
  2. 去除停用词（停用词文件cn_stopwords.txt)
*  doc2vec 抽取段落特征向量 (doc2vec_model.py)

## 具体完成记录
### 6.22
1.今日完成
  * 完成了 doc2vec 抽取特征向量()
  * 确定了文件组织形式

2.待解决问题
  * doc2vec 超参数如何确定
  * 抽取效果一般
