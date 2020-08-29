import pprint

# a = [[1,0.0256,0.01324,0.0598],[21,0.1254,0.09878,0.2465],[12,0.1248,0.0003654,0.6514]]
# sort_a = [value for index, value in sorted(enumerate(a), key=lambda a:a[1])]
# 这是一段神奇的代码，可以将多维列表a，按照首元素的大小，从小到大排列（如果要从大到小排序，只需加上reverse=True即可），而不改变每个列表后续元素的顺序，例如上述的多维列表a可以转化成：
# [[1,0.0256,0.01324,0.0598],[12,0.1248,0.0003654,0.6514],[21,0.1254,0.09878,0.2465]]
# 而enumerate（枚举）函数的作用如下：
# >>>seasons = ['Spring', 'Summer', 'Fall', 'Winter']
# >>> list(enumerate(seasons))
# [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]

# 而sorted函数的应用仅限于字典转化为元组以后，或者应用counter函数以后的情况：
# >>> myList = [('dungeon',7),('winterfell',4),('bran',9),('meelo',6)]
# >>> print sorted(myList, key=lambda x:x[1])
# [('winterfell', 4), ('meelo', 6), ('dungeon', 7), ('bran', 9)]


# 将数据存储成py文件，这样导入数据时可以维持数据的格式
def save_as_py(path, dic, content_name):  # string,dict,string
    # path是路径，dict是需要保存的数据，content_name是对应的属性变量名
    result_file = open(path, 'w', encoding="UTF-8")
    result_file.write(content_name + pprint.pformat(dic))
# a = 1
# save_as_py('../Node2vec_data/Data_a.py',a,'a_data=')


# 步骤1：依次读取emb文件，记得更改文件名字，得到字符串列表test_data1
test_data1 = []
with open('Data_mashup_tags1_20_cate_undirected.emb','r') as file:
    for line in file:
        test_data1.append(line)

# 步骤2：将test_data1里的字符列表按空格分隔开，由于第一行表示的是数据集的维度大小，不需要加进来，所以是range(1, len(test_data1)，处理后得到test_data2
test_data2 = []
for i in range(1, len(test_data1)):
    cur_list1 = list(test_data1[i].split(' '))
    test_data2.append(cur_list1)

# 步骤3：将分隔好的字符串转化成浮点型数据，由于emb数据集里面的数据不是按顺序进行排列的，为了跟后续的LDA,Doc2vec等模型接轨，需要按首元素进行排序。
test_data3 = []
for i in range(len(test_data2)):
    test_data3.append(list(map(float, test_data2[i])))
    # map(function, sequence) 会根据提供的函数对指定序列做循环映射。
# test_data2 = list(map(int, test_data2))
sort_b = [value for index, value in sorted(enumerate(test_data3), key=lambda b:b[1])]
# 这是一段神奇的代码，可以将多维列表a，按照首元素的大小，从小到大排列（如果要从大到小排序，只需加上reverse=True即可），而不改变每个列表后续元素的顺序

# 步骤4：判断长度是否符合要求，长度不足则需要不足以后再进行操作，API数据集20类数据一共是6718，mashup数据集20类数据一共是3928

# API数据构建
# data_list = []
# for i in range(6718):
#     data_list.append([i+1]+[0]*128)
#
# if len(sort_b) != 6718:
#     for i in range(6718):
#         for j in range(len(sort_b)):
#             if data_list[i][0] == int(sort_b[j][0]):
#                 data_list[i] = sort_b[j]
#
# if len(sort_b) == 6718:
#     data_list = sort_b

# mashup数据构建
data_list = []
for i in range(3928):
    data_list.append([i+1]+[0]*128)

if len(sort_b) != 3928:
    for i in range(3928):
        for j in range(len(sort_b)):
            if data_list[i][0] == int(sort_b[j][0]):
                data_list[i] = sort_b[j]

if len(sort_b) == 3928:
    data_list = sort_b

# 步骤5：由于首元素是项目的编号，最后用到的数据知识表征向量，故需要切片
fin_data_list = []
for i in range(len(data_list)):
    fin_data_list.append(data_list[i][1:])

# save_as_py('../Node2vec_data/Data_Node2_API_invoke1_20_cate.py',fin_data_list,'Node2_API_invoke1_20_cate_data=')
# save_as_py('../Node2vec_data/Data_Node2_API_tags1_20_cate.py',fin_data_list,'Node2_API_tags1_20_cate_data=')
# save_as_py('../Node2vec_data/Data_Node2_mashup_invoke1_20_cate.py',fin_data_list,'Node2_mashup_invoke1_20_cate_data=')
save_as_py('../Node2vec_data/Data_Node2_mashup_tags1_20_cate.py',fin_data_list,'Node2_mashup_tags1_20_cate_data=')
# 对于'../Node2vec_data/Data_Node2_mashup_tags1_20_cate.py'，'../'表示将数据保存到上一级目录中
