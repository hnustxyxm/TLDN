# 前20类数据提取及LDA向量获取

'''
【数据源样例】
词语1 词语2 词语3 词语4 词语5 词语6 词语7 词语8 词语9
词语1 词语2 词语3 词语4 词语5
词语1 词语2 词语3 词语4 词语5 词语6 词语7
……
一行是一篇已切好词的文本，词语之间用空格分隔
【主要参数说明】
1.n_topics：主题个数，即需要将这些文本聚成几类
2.n_iter：迭代次数
【程序输出说明】
1.doc-topic分布：即每篇文本属于每个topic的概率，比如20个topic，那么第一篇文本的doc-topic的分布就是该文本属于这20个
topic的概率（一共20个概率数字）
2.topic-word分布：即每个topic内词的分布，包含这个词的概率/权重
3.每个topic内权重最高的5个词语
4.每篇文本最可能的topic
'''

import codecs
import collections
import numpy as np
import lda
import pandas as pd
import re
import pprint
# test_data = pd.read_csv("C:\Python\laboratory_datas/Mashup_preprocess_data1.csv")
# test_data1 = test_data.values

Mashup_preprocess_data = pd.read_csv("C:\Python\laboratory_datas/Mashup_preprocess_data.csv")
Preprocess_desc = Mashup_preprocess_data['preprocess_desc'].values.tolist()

print(Preprocess_desc[0])

Mashup_data = pd.read_csv("C:\Python\laboratory_datas/Unrepeat_Mashups.csv")
# Mashup_doc2vec_data = pd.read_csv("C:\Python\laboratory_datas/Mashup_doc2vec_data.csv")

All_ID = Mashup_data['MashupName'].values
All_cate = Mashup_data['primary_category'].values
All_desc = Mashup_data['desc'].values

print(len(All_ID))
# 统计词频，统计出每个种类出现的次数
Number_cate = collections.Counter(All_cate)
# 举例：str1=['a','a','b','d']；m=collections.Counter(str1)；print(m)
# -->  Counter({'a': 2, 'b': 1, 'd': 1})

# 排序，将种类数从高到低排列
sort_cate = sorted(Number_cate.items(), key=lambda x: x[1], reverse=True)

# print(sort_cate[:20])
# [('Mapping', 1028), ('Search', 306), ('Social', 298), ('eCommerce', 292), ('Photos', 256), ('Music', 247), ('Video', 174), ('Travel', 169), ('Messaging', 131), ('Mobile', 126), ('Sports', 118), ('News Services', 113), ('Reference', 97), ('Telephony', 97), ('Blogging', 96), ('Electronic Signature', 91), ('Widgets', 81), ('Visualizations', 74), ('Real Estate', 67), ('Humor', 67)]

print(sort_cate)
# 统计项目数目总数
Mashup_cate_param = 20
Mashup_item_num = 0
for i in range(Mashup_cate_param):
    Mashup_item_num += sort_cate[i][1]


# 正则匹配,将字符串里单引号内的内容提取出来，这波很关键
def p_words(string):
    string_list = re.findall(r"\'\w*\'", string)
    return string_list


# 将数据存储成py文件，这样导入数据时可以维持数据的格式
def save_as_py(path, dic, content_name):  # string,dict,string
    # path是路径，dict是需要保存的数据，content_name是对应的属性变量名
    result_file = open(path, 'w', encoding="UTF-8")
    result_file.write(content_name + pprint.pformat(dic))

# 构建数据集
LDA_Dataset_20categories = []
Label_20categories = []
Cate_20categories = []
for i in range(Mashup_cate_param):
    for j in range(len(All_ID)):
        if All_cate[j] == sort_cate[i][0]:
            LDA_Dataset_20categories.append(p_words(Preprocess_desc[j]))
            Label_20categories.append(i)
            Cate_20categories.append(All_cate[j])

# 1.删除对主题结果影响较大，且无意义的单词，如“api”、“use”、“add”、“data”、“user”，要保存向量，一共要改三个地方，下面1.2.3.点分别进行了说明
# for lda_data in LDA_Dataset_20categories:
#     while lda_data.count("'api'"):
#         lda_data.remove("'api'")
#     while lda_data.count("'use'"):
#         lda_data.remove("'use'")
#     while lda_data.count("'user'"):
#         lda_data.remove("'user'")
#     while lda_data.count("'data'"):
#         lda_data.remove("'data'")
#     while lda_data.count("'add'"):
#         lda_data.remove("'add'")
# 2.删除掉一些词以后，保存
# save_as_py('./Data_Premashup_20_cate_filter.py',LDA_Dataset_20categories,'Premashup_20_cate_filter_data=')
# save_as_py('./Data_Labelmashup20_cate_filter.py',Label_20categories,'Labelmashup_20_cate_filter_data=')

# # 保存前20类mashup数据
save_as_py('./Data_Premashup_20_cate.py',LDA_Dataset_20categories,'Premashup_20_cate_data=')
save_as_py('./Data_Labelmashup20_cate.py',Label_20categories,'Labelmashup_20_cate_data=')

assert len(LDA_Dataset_20categories) == len(Label_20categories) == Mashup_item_num
# print(LDA_Dataset_20categories[:20])
# print(Label_20categories[1028])
# 读取已切好词的语料库所有词语，去重
for i in range(10):
    dict3 = collections.Counter(LDA_Dataset_20categories[i])
    print(dict3)
    print(Cate_20categories[i])

wordList1 = []
for i in range(len(LDA_Dataset_20categories)):
    wordList1 += (LDA_Dataset_20categories[i])
wordList = set(wordList1)
wordList = list(wordList)

print(wordList1[0])
print(len(wordList1))
print(len(wordList))

# 生成词频矩阵，一行一个文本，一列一个词语，数值等于该词语在当前文本中出现的频次
# 矩阵行数=文本总数，矩阵列数=语料库去重后词语总数
# 该矩阵是一个大的稀疏矩阵
wordMatrix = []
for eachLine2 in LDA_Dataset_20categories:
    # docWords = eachLine2.strip().split(',')
    dict1 = collections.Counter(eachLine2)
    key1 = list(dict1.keys())
    r1 = []
    for i in range(len(wordList)):
        if wordList[i] in key1:
            r1.append(dict1[wordList[i]])
        else:
            r1.append(0)
    wordMatrix.append(r1)
X = np.array(wordMatrix)    # 词频矩阵

# 模型训练
model = lda.LDA(n_topics=20, n_iter=50, random_state=1)
model.fit(X)

# doc-topic分布
print('==================doc:topic==================')
doc_topic = model.doc_topic_
print(type(doc_topic))
print(doc_topic.shape)
# print(doc_topic)    # 一行为一个doc属于每个topic的概率，每行之和为1
a = []
b = []
for i in range(10):
    a.append(max(doc_topic[i]))
    for j in range(len(doc_topic[i])):
        if doc_topic[i][j] == max(doc_topic[i]):
            b.append(j)
print(a)
print(b)
# # 保存前20类mashup的主题向量数据,注意这里要将np的array矩阵转化成list格式，其中list转array用np.array(List)，而list转array用Array.tolist
# save_as_py('./Data_LDAmashup_20_cate.py',doc_topic.tolist(),'LDAmashup_20_cate_data=')

# 3.删除对主题结果影响较大，且无意义的单词以后，训练得到的向量
# save_as_py('./Data_LDA_mashup_20_cate_filter.py',doc_topic.tolist(),'LDA_mashup_20_cate_data_filter=')

# topic-word分布
print('==================topic:word==================')
topic_word = model.topic_word_
print(type(topic_word))
print(topic_word.shape)
# print(topic_word[:, :3])    # 一行对应一个topic，即每行是一个topic及该topic下词的概率分布，每行之和为1

# 每个topic内权重最高的5个词语
n = 5
print('==================topic top' + str(n) + ' word==================')
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(wordList)[np.argsort(topic_dist)][:-(n+1):-1]
    # np.argsort可以将每一维度的数据从小到大排列，[-6:-1]表示去列表的最后五个元素
    print('*Topic {}\n-{}'.format(i, ' '.join(topic_words)))

# 每篇文本最可能的topic
print('==================doc best topic==================')
txtNums = len(codecs.open("C:\Python\laboratory_datas/Mashup_preprocess_data.csv", 'r', 'utf-8').readlines())   #文本总数
for i in range(10):
    topic_most_pr = doc_topic[i].argmax()
    print('doc: {} ,best topic: {}'.format(i, topic_most_pr))

