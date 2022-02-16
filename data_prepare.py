# jieba.set_dictionary('jieba词典.txt')
import pandas as pd
import numpy as np
import jieba
from gensim.models import Word2Vec
from tensorflow.keras import Model, preprocessing

# def not_chinese(uchar):
#     if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
#         return False
#     else:
#         return True


def load_dataset(data_url, labels_url):
    data = open(data_url).readlines()
    for i in range(len(data)):
        data[i] = data[i].strip().split()
    labels = np.loadtxt(labels_url, dtype=int, delimiter=',')
    return data, labels
#
#
# sw_file = open(r'哈工大停用词表.txt').readlines()
# stopwords = [word.strip() for word in sw_file]
#
# dataframe = pd.read_csv(r'asap-master/data/test.csv')
# reviews = dataframe['review'].tolist()
# file = open(r'data/test_data.txt', 'w')
# for review in reviews:
#     words = jieba.cut(review, cut_all=False)
#     for word in words:
#         if not_chinese(word):
#             continue
#         elif word not in stopwords and len(word) > 1:
#             file.write(word+' ')
#     file.write('\n')
# file.close()
#
# labels = dataframe.iloc[:, 3:].to_numpy()
# np.savetxt(r'data/test_labels.txt', labels, delimiter=',', fmt='%d')

# labels = np.loadtxt(r'test_labels.txt', delimiter=',', dtype=int)
# print(labels)


file1 = open(r'data/train_data.txt').readlines()
for i in range(len(file1)):
    file1[i] = file1[i].strip().split()
file2 = open(r'data/test_data.txt').readlines()
for i in range(len(file2)):
    file2[i] = file2[i].strip().split()
sentences = file1 + file2


# w2vmodel = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=5)
# vocab = list(w2vmodel.wv.key_to_index.keys())
# vocab.insert(0, 'pad')
# file3 = open(r'data/Vocabulary.txt', 'w')
# for word in vocab:
#     file3.write(word+' ')
# file3.close()


test_data, test_labels = load_dataset(r'test_data.txt', 'test_labels.txt')
file = open(r'data/Vocabulary.txt').readline()
vocab = file.strip().split()
#
#
# sen_len = 100
# wv_len = 100
#
# sen_idx = []
# for i in range(len(test_data)):
#     tmp = []
#     for word in test_data[i]:
#         if word not in vocab:
#             continue
#         tmp.append(vocab.index(word))
#     sen_idx.append(tmp)
#
# a = preprocessing.sequence.pad_sequences(sen_idx, value=0, padding='post', maxlen=100)
# np.savetxt(r'data/paded_sen_idx.txt', a, fmt='%d', delimiter=' ')
paded_sen_idx = np.loadtxt(r'data/paded_sen_idx.txt', dtype=int, delimiter=' ')







