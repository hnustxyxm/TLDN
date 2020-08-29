import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# !!!注意：data_path的文件格式必须是：CSV(逗号分隔)(*.csv)

# mashup_path = "C:\Python\laboratory_datas/Unrepeat_Mashups.csv"
mashup_path = "C:\Python\laboratory_datas/APIs.csv"
mashup = pd.read_csv(mashup_path, dtype={'desc':str}, encoding='iso-8859-1')
mashup = mashup[['desc']].astype('str')    #将节点列变成字符串格式
# !!!注意：当文件里面出现类型不一样的内容时，可以使用dtype和astype来设置你读取相应内容的类型

mashup_desc = mashup.desc

# 此处需要改变对应的目录

texts_tokenized = [[word.lower() for word in word_tokenize(document)] for document in mashup_desc]

# 定义一个标点符号的词典，用这个词典来过滤标点符号
english_punctuations = [',','.',':',';','?','!','(',')','[',']','@','&','#','%','$','{','}','--','-','"','``',"'"]
# 1.单引号中可以使用双引号,中间的会当作字符串输出2.双引号中可以使用单引号,中间的会当作字符串输出3.三单引号和三双引号中间的字符串在输出时保持原来的格式。
texts_filtered = [[word for word in document if not word in english_punctuations] for document in texts_tokenized]

# 接下来将这些英文单词词干化，词干化可以提取不同语态及各种后缀的词干
st = PorterStemmer()
texts_stemmed = [[st.stem(word) for word in document] for document in texts_filtered]


# 去停用词，用nltk带有的停用词表
english_stopwords = stopwords.words('english')
texts_stemmed = [[word for word in document if not word in english_stopwords] for document in texts_stemmed]
print(len(texts_stemmed))

# 将mashup对应的ID和处理好的描述信息建立成字典形式
# mashup_data = pd.read_csv("C:\Python\laboratory_datas/Unrepeat_Mashups.csv")
# Allmashupdata = mashup_data['MashupName'].values
mashup_data = pd.read_csv("C:\Python\laboratory_datas/APIs.csv")
Allmashupdata = mashup_data['APIName'].values

fin_data = zip(Allmashupdata,texts_stemmed)

# Mashup_preprocess_data = pd.DataFrame(list(fin_data),columns=['MahupName','preprocess_desc'])
# Mashup_preprocess_data.to_csv('C:\Python\laboratory_datas/Mashup_preprocess_data1.csv',encoding='utf-8',index=0)

API_preprocess_data = pd.DataFrame(list(fin_data),columns=['APIName','preprocess_desc'])
API_preprocess_data.to_csv('C:\Python\laboratory_datas/API_preprocess_data.csv',encoding='utf-8',index=0)
# header=0表示不保存列名；index=0表示不保留行索引

'''
# dict_data =[]
# for x in Allmashupdata:
#     if x not in dict_data:
#         dict_data.append(x)
#     else:
#         print(x)
dict_data = dict(fin_data)
print(len(Allmashupdata))
print(len(dict_data))

# 保存预处理好的数据
# Mashup_preprocess_data = pd.DataFrame.from_dict(data=dict_data, orient='index')
# 这里要将pd.DataFrame(data=dict_data)改成pd.DataFrame.from_dict(data=dict_data, orient='index')，不然就会报错ValueError: arrays must all be same length
# 这是因为在这种操作方法中，每个单词都会被当成一列，而每个Mashup的描述文本的长度不一样，所以要引入pd.DataFrame.from_dict
# orient:{‘columns’，‘index’}，默认’列’数据的“方向”。如果传递的dict的键应该是结果DataFrame的列，则传递’columns’（默认值）。否则，如果键应该是行，则传递’index’

# 但是，如果将所有的Key（Mashup的ID）当成一列，所有的Value（描述文本单词）当成一列就不会出现长短不一的现象，从而就可以继续用pd.DataFrame
k = list(dict_data.keys())
v = list(dict_data.values())
Mashup_preprocess_data = pd.DataFrame(list(zip(k,v)),columns=['MahupName','preprocess_desc'])
# columns参数可以指定表头

Mashup_preprocess_data.to_csv('C:\Python\laboratory_datas/Mashup_preprocess_data.csv',encoding='utf-8',index=0)
# header=0表示不保存列名；index=0表示不保留行索引
'''
