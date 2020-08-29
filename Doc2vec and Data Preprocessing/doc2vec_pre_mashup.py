# 计算所有数据的Doc2vec向量
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import sys
import os
import gensim
import numpy as np
from gensim.models import Doc2Vec

def mashup_preprosess(data_path):
    # !!!注意：data_path的文件格式必须是：CSV(逗号分隔)(*.csv)

    mashup_path = data_path
    mashup = pd.read_csv(mashup_path, dtype={'desc':str}, encoding='iso-8859-1')
    mashup = mashup[['desc']].astype('str')    #将节点列变成字符串格式
    # !!!注意：当文件里面出现类型不一样的内容时，可以使用dtype和astype来设置你读取相应内容的类型

    mashup_desc = mashup.desc

    # 此处需要改变对应的目录

    texts_tokenized = [[word.lower() for word in word_tokenize(document)] for document in mashup_desc]

    # 定义一个标点符号的词典，用这个词典来过滤标点符号
    english_punctuations = [',','.',':',';','?','!','(',')','[',']','@','&','#','%','$','{','}','--','-','``','"',"'"]
    texts_filtered = [[word for word in document if not word in english_punctuations] for document in texts_tokenized]

    # 接下来将这些英文单词词干化，词干化可以提取不同语态及各种后缀的词干
    st = PorterStemmer()
    texts_stemmed = [[st.stem(word) for word in document] for document in texts_filtered]


    # 去停用词，用nltk带有的停用词表
    english_stopwords = stopwords.words('english')
    texts_stemmed = [[word for word in document if not word in english_stopwords] for document in texts_stemmed]

    return texts_stemmed

def doc2vec(data_path):
    # 引入doc2vec
    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = os.path.split(curPath)[0]
    sys.path.append(rootPath)
    # 引入日志配置
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # 加载数据
    documents = []
    # 使用第一列当做每个句子的“标签”，标签和每个句子是一一对应的
    count = 0
    # pd.read_csv("C:\Python\\anaconda/Api_Info_all.csv",encoding='utf-8')
    # content_x = df.Description.values

    # 使用api_preprosess()进行预处理
    words = mashup_preprosess(data_path)
    for word in words:
        #添加到数组中
        documents.append(gensim.models.doc2vec.TaggedDocument(word, [str(count)]))
        count += 1
    # 模型训练
    model = Doc2Vec(documents, dm=1, size=128, window=8, min_count=5, workers=4)
    # 参数说明：dm（{1 ，0} ，任选的） -定义的训练算法。如果dm = 1，则使用“分布式内存”（PV-DM）。否则，将使用分布式单词袋（PV-DBOW）。
    # size（int ，可选）–特征向量的维数
    # window（int ，可选）–句子中当前单词和预测单词之间的最大距离，即窗口定义
    # min_count（int ，optional）–忽略总频率低于此频率的所有单词。
    # worker（int ，可选）–使用这些许多worker线程来训练模型（=使用多核计算机进行更快的训练）

    # 保存模型
    model.save('doc2vec.model')
    # 保存向量
    corpus = model.docvecs
    # np.savetxt("xy.txt", corpus)
    vectors = []
    for i in range(len(corpus)):
        vector = corpus[i]
        vectors.append(vector)
    data_x = np.array(vectors)

    mashup_data = pd.read_csv("C:\Python\laboratory_datas/Unrepeat_Mashups.csv")
    Allmashupdata = mashup_data['MashupName'].values

    fin_data = zip(Allmashupdata,data_x)
    dict_data = dict(fin_data)

    return dict_data

def cosine_similarity(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return dot_product / ((normA**0.5) * (normB**0.5))

data_x = doc2vec("C:\Python\laboratory_datas/Unrepeat_Mashups.csv")
# api_simi = api_similar(data_x)
# cos_simi = cosine_similarity(data_x['Opuss'],data_x['Opuss'])
# print(np.shape(data_x))
print(len(data_x))
# print(cos_simi)

# 但是，如果将所有的Key（Mashup的ID）当成一列，所有的Value（描述文本单词）当成一列就不会出现长短不一的现象，从而就可以继续用pd.DataFrame
k = list(data_x.keys())
v = list(data_x.values())
print(len(list(zip(k,v))))
Mashup_preprocess_data = pd.DataFrame(list(zip(k,v)),columns=['MahupName','Doc2vec_vector'])

# Mashup_preprocess_data = pd.DataFrame(data=data_x).T
Mashup_preprocess_data.to_csv('C:\Python\laboratory_datas/Mashup_doc2vec_data.csv',encoding='utf-8', index=0)
# header=0表示不保存列名；index=0表示不保留行索引
