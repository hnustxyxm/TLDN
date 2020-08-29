from sklearn.model_selection import train_test_split
import torch
from Data_Labelmashup20_cate import Labelmashup_20_cate_data
from Data_LDA_mashup_20_cate import LDAmashup_20_cate_data
# Data_LDA_mashup_20_cate里面的数据命名出现了问题，少了一根_,导入该数据的时候需要注意

from Data_Doc2mashup_20_cate import Doc2mashup_20_cate_data
from Data_LDA_mashup_20_cate_filter import LDA_mashup_20_cate_data_filter
from Data_TFIDF_mashup_20_cate import TFIDF_mashup_20_cate_data
from Data_Node2_mashup_invoke1_20_cate import Node2_mashup_invoke1_20_cate_data
from Data_Node2_mashup_tags1_20_cate import Node2_mashup_tags1_20_cate_data
import pprint
import random

# 将数据存储成py文件，这样导入数据时可以维持数据的格式
def save_as_py(path, dic, content_name):  # string,dict,string
    # path是路径，dict是需要保存的数据，content_name是对应的属性变量名
    result_file = open(path, 'w', encoding="UTF-8")
    result_file.write(content_name + pprint.pformat(dic))


# 联合训练
x0 = LDAmashup_20_cate_data  # Data_LDA_mashup_20_cate里面的数据命名出现了问题，少了一根_,导入该数据的时候需要注意填LDAmashup_20_cate_data
x1 = Doc2mashup_20_cate_data
x2 = TFIDF_mashup_20_cate_data
x3 = Node2_mashup_invoke1_20_cate_data
x4 = Node2_mashup_tags1_20_cate_data
y = Labelmashup_20_cate_data

LDA_Doc2_TFIDF_invoke_tags = []
for i in range(len(y)):
    LDA_Doc2_TFIDF_invoke_tags.append(x0[i]+x1[i]+x2[i]+x3[i]+x4[i])

Doc2_TFIDF_invoke_tags = []
for i in range(len(y)):
    Doc2_TFIDF_invoke_tags.append(x1[i]+x2[i]+x3[i]+x4[i])

LDA_TFIDF_invoke_tags = []
for i in range(len(y)):
    LDA_TFIDF_invoke_tags.append(x0[i]+x2[i]+x3[i]+x4[i])

LDA_Doc2_invoke_tags = []
for i in range(len(y)):
    LDA_Doc2_invoke_tags.append(x0[i]+x1[i]+x3[i]+x4[i])

LDA_Doc2_TFIDF_tags = []
for i in range(len(y)):
    LDA_Doc2_TFIDF_tags.append(x0[i]+x1[i]+x2[i]+x4[i])

LDA_Doc2_TFIDF_invoke = []
for i in range(len(y)):
    LDA_Doc2_TFIDF_invoke.append(x0[i]+x1[i]+x2[i]+x3[i])

Data_list = [LDA_Doc2_TFIDF_invoke_tags,Doc2_TFIDF_invoke_tags,LDA_TFIDF_invoke_tags,LDA_Doc2_invoke_tags,LDA_Doc2_TFIDF_tags,LDA_Doc2_TFIDF_invoke]
output_list_train_X1 = ['Data_mashup_X_train_01234.py','Data_mashup_X_train_1234.py','Data_mashup_X_train_0234.py',
                        'Data_mashup_X_train_0134.py','Data_mashup_X_train_0124.py','Data_mashup_X_train_0123.py']
output_list_train_X2 = ['mashup_X_train_01234_data=','mashup_X_train_1234_data=','mashup_X_train_0234_data=',
                        'mashup_X_train_0134_data=','mashup_X_train_0124_data=','mashup_X_train_0123_data=']
output_list_test_X1 = ['Data_mashup_X_test_01234.py','Data_mashup_X_test_1234.py','Data_mashup_X_test_0234.py',
                       'Data_mashup_X_test_0134.py','Data_mashup_X_test_0124.py','Data_mashup_X_test_0123.py']
output_list_test_X2 = ['mashup_X_test_01234_data=','mashup_X_test_1234_data=','mashup_X_test_0234_data=',
                       'mashup_X_test_0134_data=','mashup_X_test_0124_data=','mashup_X_test_0123_data=']
output_list_train_Y1 = ['Data_mashup_Y_train_01234.py','Data_mashup_Y_train_1234.py','Data_mashup_Y_train_0234.py',
                        'Data_mashup_Y_train_0134.py','Data_mashup_Y_train_0124.py','Data_mashup_Y_train_0123.py']
output_list_train_Y2 = ['mashup_Y_train_01234_data=','mashup_Y_train_1234_data=','mashup_Y_train_0234_data=',
                        'mashup_Y_train_0134_data=','mashup_Y_train_0124_data=','mashup_Y_train_0123_data=']
output_list_test_Y1 = ['Data_mashup_Y_test_01234.py','Data_mashup_Y_test_1234.py','Data_mashup_Y_test_0234.py',
                       'Data_mashup_Y_test_0134.py','Data_mashup_Y_test_0124.py','Data_mashup_Y_test_0123.py']
output_list_test_Y2 = ['mashup_Y_test_01234_data=','mashup_Y_test_1234_data=','mashup_Y_test_0234_data=',
                       'mashup_Y_test_0134_data=','mashup_Y_test_0124_data=','mashup_Y_test_0123_data=']

# 注意：0代表LDA表征向量，1代表Doc2vec表征向量，2代表TFIDF表征向量，3代表服务调用结构表征向量，4代表服务标签结构表征向量。
# 缺少的数字就表示对应缺少的表征向量

for i in range(len(Data_list)): 
    X_train, X_test, Y_train, Y_test = train_test_split(Data_list[i], y, test_size=0.30, random_state=26)
    # 此处调用了sklearn.model_selection里面自动进行训练集与测试集划分的包train_test_split，详情可百度搜索
    # test_size：样本占比，如果是整数的话就是样本的数量
    # random_state：是随机数的种子。
    # 随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
    save_as_py(output_list_train_X1[i],X_train,output_list_train_X2[i])
    save_as_py(output_list_test_X1[i],X_test,output_list_test_X2[i])
    save_as_py(output_list_train_Y1[i],Y_train,output_list_train_Y2[i])
    save_as_py(output_list_test_Y1[i],Y_test,output_list_test_Y2[i])
    print('第'+str(i)+'次遍历')
