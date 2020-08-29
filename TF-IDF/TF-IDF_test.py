from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# 将文本中的词转换成词频矩阵
vectorizer = CountVectorizer()
print(vectorizer)
texts = []
# for i in range(len(Premashup_20_cate_data)):
#     texts.append(" ".join(str(i) for i in Premashup_20_cate_data))
# print(texts)
# 计算某个词出现的次数
X = vectorizer.fit_transform(texts)
print(type(X),X)
# 获取词袋中所有文本关键词
word = vectorizer.get_feature_names()
print(word)
# 查看词频结果
print(X.toarray())

# 类调用
transformer = TfidfTransformer()
print(transformer)
# 将词频矩阵统计成TF-IDF值
Tf_Idf = transformer.fit_transform(X)
# 查看数据结构Tf_Idf[i][j]表示i类文本中tf-idf权重
print(Tf_Idf.toarray())
