from Data_Premashup_20_cate import Premashup_20_cate_data
import sys
import os
import gensim
import numpy as np
from gensim.models import Doc2Vec
import pprint

# 将数据存储成py文件，这样导入数据时可以维持数据的格式
def save_as_py(path, dic, content_name):  # string,dict,string
    # path是路径，dict是需要保存的数据，content_name是对应的属性变量名
    result_file = open(path, 'w', encoding="UTF-8")
    result_file.write(content_name + pprint.pformat(dic))

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
words = Premashup_20_cate_data
for word in words:
    # 添加到数组中
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
    vectors.append(vector.tolist())
save_as_py('./Data_Doc2mashup_20_cate.py',vectors,'Doc2mashup_20_cate_data=')
