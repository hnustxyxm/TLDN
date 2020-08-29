from Data_Premashup_20_cate import Premashup_20_cate_data
from Data_PreAPI_20_cate import PreAPI_20_cate_data
from collections import defaultdict
import math
import operator
import pprint

"""
函数说明：特征选择TF-IDF算法
Parameters:
     list_words:词列表
Returns:
     dict_feature_select:特征选择词字典
"""


def save_as_py(path, dic, content_name):  # string,dict,string
    # path是路径，dict是需要保存的数据，content_name是对应的属性变量名
    result_file = open(path, 'w', encoding="UTF-8")
    result_file.write(content_name + pprint.pformat(dic))


def feature_select(list_words):
    # 总词频统计
    doc_frequency = defaultdict(int)
    for word_list in list_words:
        for i in word_list:
            doc_frequency[i] += 1

    # 计算每个词的TF值
    word_tf = {}  # 存储没个词的tf值
    for i in doc_frequency:
        word_tf[i] = doc_frequency[i]/sum(doc_frequency.values())

    # 计算每个词的IDF值
    doc_num = len(list_words)
    word_idf = {}  # 存储每个词的idf值
    word_doc = defaultdict(int)  # 存储包含该词的文档数
    for i in doc_frequency:
        for j in list_words:
            if i in j:
                word_doc[i] += 1
    for i in doc_frequency:
        word_idf[i] = math.log(doc_num/(word_doc[i]+1))

    # 计算每个词的TF*IDF的值
    word_tf_idf = {}
    for i in doc_frequency:
        word_tf_idf[i] = word_tf[i]*word_idf[i]

    # 对字典按值由大到小排序
    dict_feature_select = sorted(word_tf_idf.items(),key=operator.itemgetter(1),reverse=True)
    return dict_feature_select

if __name__ == '__main__':
    # data_list = Premashup_20_cate_data  # 加载mashup数据
    data_list = PreAPI_20_cate_data  # 加载API数据
    features = feature_select(data_list)  # 所有词的TF-IDF值
    print(features)
    print(len(features))
    print(type(features))
    print(features[0][0])
    print(features[0][1])
    # print(Premashup_20_cate_data[0][0])

    # # 构建文档向量列表，由于20类mashup数据集中一共有3928个Mashup，且包含的文本中，最多是196个单词，所以需要构建3928*200的列表
    # TF_IDF_list = []
    # for i in range(3928):
    #     TF_IDF_list.append([0]*200)

    # 构建文档向量列表，由于20类API数据集中一共有6718个API，且包含的文本中，最多是209个单词，为了统一，我们同样构建6718*210的列表
    TF_IDF_list = []
    for i in range(6718):
        TF_IDF_list.append([0]*210)

    for i in range(len(PreAPI_20_cate_data)):
        for j in range(len(PreAPI_20_cate_data[i])):
            for k in range(len(features)):
                if PreAPI_20_cate_data[i][j] == features[k][0]:
                    TF_IDF_list[i][j] = features[k][1]

    # print(TF_IDF_list[0])
    #
    # save_as_py('./Data_TFIDF_mashup_20_cate.py',TF_IDF_list,'TFIDF_mashup_20_cate_data=')
    save_as_py('./Data_TFIDF_API_20_cate.py',TF_IDF_list,'TFIDF_API_20_cate_data=')
