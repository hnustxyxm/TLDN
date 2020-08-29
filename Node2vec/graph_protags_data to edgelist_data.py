import pandas as pd
import collections
from Data_API_tags1_20_cate_graph import API_tags1_20_cate_graph_data
from Data_API_tags2_20_cate_graph import API_tags2_20_cate_graph_data
from Data_mashup_tags1_20_cate_graph import mashup_tags1_20_cate_graph_data
from Data_mashup_tags2_20_cate_graph import mashup_tags2_20_cate_graph_data

Mashup_data = pd.read_csv("C:\Python\laboratory_datas/Unrepeat_Mashups.csv")
API_data = pd.read_csv("C:\Python\laboratory_datas/APIs.csv")
# Mashup_doc2vec_data = pd.read_csv("C:\Python\laboratory_datas/Mashup_doc2vec_data.csv")

All_API_cate = API_data['primary_category'].values
All_Mashup_cate = Mashup_data['primary_category'].values

# 统计词频，统计出每个种类出现的次数
Number_Mashup_cate = collections.Counter(All_Mashup_cate)
Number_API_cate = collections.Counter(All_API_cate)
# 举例：str1=['a','a','b','d']；m=collections.Counter(str1)；print(m)
# -->  Counter({'a': 2, 'b': 1, 'd': 1})

# 排序，将种类数从高到低排列
sort_Mashup_cate = sorted(Number_Mashup_cate.items(), key=lambda x: x[1], reverse=True)
sort_API_cate = sorted(Number_API_cate.items(), key=lambda x: x[1], reverse=True)

# 统计项目数目总数
Mashup_cate_param = 20
Mashup_item_num = 0
Mashup_cate_list = []
for i in range(Mashup_cate_param):
    Mashup_item_num += sort_Mashup_cate[i][1]

API_cate_param = 20
API_item_num = 0
API_cate_list = []
for i in range(API_cate_param):
    API_item_num += sort_API_cate[i][1]

# 对mashup的tag数据集进行处理，从而减少计算量
mashup_tags1 = mashup_tags1_20_cate_graph_data
mashup_tags2 = []
assert Mashup_item_num == len(mashup_tags1)
cur_num = 1
for i in range(20):
    for j in range(cur_num, cur_num + sort_Mashup_cate[i][1]-1):
        del mashup_tags1[j][1:sort_Mashup_cate[i][1]]
    # del mashup_tags2[cur_num, cur_num + sort_Mashup_cate[i][1]-1]
    for k in range(3):
        print(mashup_tags1[cur_num-1+k])
    # print(mashup_tags2[cur_num-1])
    cur_num += sort_Mashup_cate[i][1]

Amount_mashup_cate = 0
for i in range(20):
    cur_list5 = list(range(Amount_mashup_cate + 1,Amount_mashup_cate + sort_Mashup_cate[i][1] + 1))
    # 生成（1-种类数）的列表，这里需要用到Amount_cate，因为序号一直在发生叠加变化，这波操作也很关键
    mashup_tags2.append(cur_list5)
    Amount_mashup_cate += sort_Mashup_cate[i][1]  # 更新种类的数目

# 对API的tag数据集进行处理，从而减少计算量
API_tags1 = API_tags1_20_cate_graph_data
API_tags2 = []
assert API_item_num == len(API_tags1)
cur_num = 1
for i in range(20):
    for j in range(cur_num, cur_num + sort_API_cate[i][1]-1):
        del API_tags1[j][1:sort_API_cate[i][1]]
    # del API_tags2[cur_num, cur_num + sort_API_cate[i][1]-1]
    for k in range(3):
        print(API_tags1[cur_num-1+k])
    # print(API_tags2[cur_num-1])
    cur_num += sort_API_cate[i][1]

Amount_API_cate = 0
for i in range(20):
    cur_list5 = list(range(Amount_API_cate + 1,Amount_API_cate + sort_API_cate[i][1] + 1))
    # 生成（1-种类数）的列表，这里需要用到Amount_cate，因为序号一直在发生叠加变化，这波操作也很关键
    API_tags2.append(cur_list5)
    Amount_API_cate += sort_API_cate[i][1]  # 更新种类的数目


input_list = [API_tags1,API_tags2,mashup_tags1,mashup_tags2]

output_list = ['Data_API_tags1_20_cate_undirected.edgelist','Data_API_tags2_20_cate_undirected.edgelist',
               'Data_mashup_tags1_20_cate_undirected.edgelist','Data_mashup_tags2_20_cate_undirected.edgelist']

for k in range(len(input_list)):
    network = input_list[k]
    # print(network[2332])
    # network = [[1],[2,1,3],[3,1,2,4],[5],[6]]
    # test_list = network

    test_list = []
    for i in range(len(network)):
        for j in range(1,len(network[i])):  # 没有边的节点就不需要进行训练，故这里从1开始
            if [network[i][0],network[i][j]] not in test_list and [network[i][j], network[i][0]] not in test_list:
                # 此判断语句用来生成无向图，若需生成有向图，可直接删除掉此此判断语句
                test_list.append([network[i][0],network[i][j]])

    # print(test_list)
    # Data_mashup_tags1_20_cate
    with open(output_list[k], "w", encoding="utf-8") as f:
        for member_content in test_list:
            for members in member_content:
                members = str(members)
                f.write(members + '\t')
            f.write('\n')
