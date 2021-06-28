import os
import gensim
import collections


SEG_FILE_PATH = "./files/seg_files"
MODEL_PATH = ".\\files\\model_file\\doc2vec_temp1.model"
# 训练模型超参
VECTOR_SIZE = 50
MIN_COUNT = 2
EPOCHS = 100


def read_document(data_root=SEG_FILE_PATH):
    for file_name in os.listdir(data_root):
        file_data = []
        with open(os.path.join(data_root, file_name), 'r', encoding='utf-8') as seg_file:
            # 以段作为单独的文章
            for i, line in enumerate(seg_file):
                tag = '%s_%s' % (file_name, i)
                tokens = line.split()
                yield gensim.models.doc2vec.TaggedDocument(tokens, [tag])


def train_save_mode(train_corpus, vector_size, min_count, epochs, model_path):
    print("训练模型中.....")
    doc2vec_model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs, workers=8)
    doc2vec_model.build_vocab(train_corpus)
    doc2vec_model.train(train_corpus, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
    # 将模型存储到指定位置
    doc2vec_model.save(model_path)
    return doc2vec_model


def evaluate_my_doc2vec(aim_model, train_corpus, test_corpus):
    ranks = []
    second_ranks = []
    print("evaluate")
    for index in range(len(train_corpus)):
        inferred_vector = aim_model.infer_vector(train_corpus[index].words)
        sims = aim_model.dv.most_similar([inferred_vector], topn=2)
        tempIds = [sim_id for sim_id, sim in sims]
        if train_corpus[index].tags[0] not in tempIds:
            rank = -1
        else:
            rank = tempIds.index(train_corpus[index].tags[0])
        ranks.append(rank)
        second_ranks.append(sims[1])
    counter = collections.Counter(ranks)
    print(counter)


def get_most_similar(aim_model, aim_corpus):
    tag_words_dic = {}
    # 首先生成字典,便于通过tag查询分词列表
    for sentence in aim_corpus:
        tag_words_dic[sentence.tags[0]] = sentence.words

    for sentence in aim_corpus:
        print(sentence.words)
        inferred_vector = aim_model.infer_vector(sentence.words)
        sims = aim_model.dv.most_similar([inferred_vector], topn=2)
        print('most similar:«%s»\n' % tag_words_dic[sims[0][0]])
        print('second similar:«%s»\n' % tag_words_dic[sims[1][0]])


def get_by_tag(aim_corpus, tag):
    for sentence in aim_corpus:
        if tag in sentence.tags:
            return sentence


def get_doc2vec(vector_size, min_count, epochs, model_path, input_path, is_load):
    if is_load:
        doc2vec_model = gensim.models.doc2vec.Doc2Vec.load(model_path)
    else:
        train_corpus = list(read_document(data_root=input_path))
        doc2vec_model = train_save_mode(train_corpus, vector_size, min_count, epochs, model_path)
    return doc2vec_model


def get_doc_vec(doc_data, trained_model):
    result = []
    for doc in doc_data:
        doc_vec = []
        for sentence in doc:
            doc_vec.append(trained_model.infer_vector(sentence))
        result.append(doc_vec)
    return result


if __name__ == "__main__":
    train_data = list(read_document())
    model = get_doc2vec(VECTOR_SIZE, MIN_COUNT, EPOCHS, MODEL_PATH, SEG_FILE_PATH, 0)
    evaluate_my_doc2vec(model, train_data, train_data)
