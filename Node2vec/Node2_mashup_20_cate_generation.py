# 构建结构图和属性图

import pandas as pd
from collections import Counter
import pprint
from text_preprocess import text_preprosessing

mashup_data = pd.read_csv("C:\Python\laboratory_datas/Unrepeat_Mashups.csv")
AllMashup = mashup_data['MashupName'].values
All_cate = mashup_data['primary_category'].values
All_desc = mashup_data['desc'].values
MemberAPIs = mashup_data['MemberAPIs'].values
All_tags = mashup_data['tags'].values
print(len(All_cate))
# Non_repeat_Mashup = set(AllMashup)
# print(len(Non_repeat_Mashup))
# print(Non_repeat_Mashup)
# 统计词频
Number_cate = Counter(All_cate)
# 举例：str1=['a','a','b','d']；m=collections.Counter(str1)；print(m)
# -->  Counter({'a': 2, 'b': 1, 'd': 1})
# 排序
sort_cate = sorted(Number_cate.items(), key=lambda x: x[1], reverse=True)
print(sort_cate)
# key指定一个接收一个参数的函数，这个函数用于从每个元素中提取一个用于比较的关键字。默认值为None。key=lambda x: x[1]表示按照Number_cate中的第二个元素进行排序
#
# lambda函数也叫匿名函数，即，函数没有具体的名称。先来看一个最简单例子：
# def f(x):
#   return x**2
#   print f(4)
# Python中使用lambda的话，写成这样
# g = lambda x : x**2
# print g(4)
#
# reverse = True 降序 ， reverse = False 升序（默认）

cate_list = []
member_api = []
tags_list = []
num = 1
for i in range(20):  # 括号里的数字决定了分为多少类
    cur_member_api = []
    cur_tags_list = []
    for j in range(len(All_cate)):
        if All_cate[j] == sort_cate[i][0]:
            # 提取对应的memberAPI，以'@@@'分开，并将每个字符串的空格消除掉，为构建结构邻接表作准备
            cur_list1 = []
            cur_list2 = MemberAPIs[j].split('@@@')
            for n in range(len(cur_list2)):
                cur_list1.append(cur_list2[n].strip())
            cur_member_api.append(cur_list1)

            # 提取对应的tags，以'###'分开，并将每个字符串的空格消除掉，为构建结构邻接表作准备
            cur_list3 = []
            cur_list4 = All_tags[j].split('###')
            for m in range(len(cur_list4)):
                cur_list3.append(cur_list4[m].strip())
            cur_tags_list.append(cur_list3)

    for k in range(sort_cate[i][1]):
        cate_list.append([num+k,i])
        member_api.append([num+k,cur_member_api[k]])
        tags_list.append([num+k,cur_tags_list[k]])
    num += sort_cate[i][1]

print(tags_list)
print(member_api[0],member_api[9],member_api[14],member_api[36],member_api[3721],member_api[3722])
print(tags_list[0],tags_list[9],tags_list[1601],tags_list[3835],tags_list[3850],tags_list[3851])

# 构建mashup的调用关系结构图列表
structure_invoke_list = []
for i in range(len(member_api)):
    structure_invoke_list.append([i+1])
# print(structure_list)

for i in range(len(member_api)):
    for j in range(len(member_api)):
        for k in range(len(member_api[i][1])):
            if member_api[i][1][k] in member_api[j][1] and i != j:
                structure_invoke_list[i].append(member_api[j][0])

# 此时structure_invoke_list里面只要同时有调用一个API就有一条边，但是真是数据集中存在同时包含两个及以上的情况，所以需要进行一些操作，具体来说有两种invoke边定义情况：
# 1、只要同时调用一个API就认为有边
# 2、同时调用两个及以上的API才认为有一条边
# 因此，我们可以对structure_tags_list进行三种操作：

# 操作1
structure_invoke1_list = []
for i in range(len(structure_invoke_list)):
    structure_invoke1_list.append(list(set(structure_invoke_list[i])))
    structure_invoke1_list[i].sort(key=structure_invoke_list[i].index)  # 这一步操作可以保证不改变原来元素的顺序

# 操作2，可以借用counter函数
structure_invoke2_list = []
for i in range(len(structure_invoke_list)):
    structure_invoke2_list.append([i+1])

invoke_count_list = []
for i in range(len(structure_invoke_list)):
    invoke_count_list.append(list(Counter(structure_invoke_list[i]).items()))
    # Counter处理完以后是Counter类型，.items()可以将该类型转化为字典类型,最终通过list转化为列表类型

for j in range(len(structure_invoke_list)):
    for k in range(len(invoke_count_list[j])):
        if invoke_count_list[j][k][1] >= 2:
            structure_invoke2_list[j].append(invoke_count_list[j][k][0])


# 构建mashup的tag共用关系结构图列表
structure_tags_list = []
for i in range(len(tags_list)):
    structure_tags_list.append([i+1])
# print(structure_list)

for i in range(len(tags_list)):
    for j in range(len(tags_list)):
        for k in range(len(tags_list[i][1])):
            if tags_list[i][1][k] in tags_list[j][1] and i != j:
                structure_tags_list[i].append(tags_list[j][0])


# print(len(structure_list))
print(structure_invoke_list[0])
print(structure_tags_list[0])
print(Counter(structure_invoke_list[0]))
print(Counter(structure_tags_list[0]))
print(len(structure_invoke_list))
print(len(structure_tags_list))

# 此时structure_tags_list里面只要同时有包含一个tag就有一条边，但是真是数据集中存在同时包含两个及以上的情况，所以需要进行一些操作，具体来说有三种tag边定义情况：
# 1、只要同时包含一个tag就认为有边
# 2、把最主要的那个tag（即cate）作为边的评判依据
# 3、同时包含两个及以上的tag才认为有一条边，在本数据集中，同时含有两个以上tags的数据较少
# 因此，我们可以对structure_tags_list进行三种操作：

# 操作1：
structure_tags1_list = []
for i in range(len(structure_tags_list)):
    structure_tags1_list.append(list(set(structure_tags_list[i])))
    structure_tags1_list[i].sort(key=structure_tags_list[i].index)  # 这一步操作可以保证不改变原来元素的顺序

# 操作2：
structure_tags2_list = []
Amount_cate = 0
for i in range(20):
    for j in range(sort_cate[i][1]):
        # 这里是把j当作调整插入位置的索引，所以此处不需要引入Amount_cate
        cur_list5 = list(range(Amount_cate + 1,Amount_cate + sort_cate[i][1] + 1))
        # 生成（1-种类数）的列表，这里需要用到Amount_cate，因为序号一直在发生叠加变化，这波操作也很关键
        num1 = cur_list5.pop(j)
        cur_list5.insert(0,num1)  # 弹出对应位置的元素并插入到列表首位
        structure_tags2_list.append(cur_list5)
    Amount_cate += sort_cate[i][1]  # 更新种类的数目
assert Amount_cate == len(structure_tags_list) == len(structure_tags2_list) == len(structure_tags1_list)

# 操作3，可以借用counter函数
structure_tags3_list = []
for i in range(len(structure_tags_list)):
    structure_tags3_list.append([i+1])

tags_count_list = []
for i in range(len(structure_tags_list)):
    tags_count_list.append(list(Counter(structure_tags_list[i]).items()))

for j in range(len(structure_tags_list)):
    for k in range(len(tags_count_list[j])):
        if tags_count_list[j][k][1] >= 2:
            structure_tags3_list[j].append(tags_count_list[j][k][0])

num2 = 0
for i in range(len(structure_tags1_list)):
    if structure_tags1_list[i] != structure_tags2_list[i]:
        num2 += 1
print(num2)


# 将数据存储成py文件，这样导入数据时可以维持数据的格式
def save_as_py(path, dic, content_name):  # string,dict,string
    # path是路径，dict是需要保存的数据，content_name是对应的属性变量名
    result_file = open(path, 'w', encoding="UTF-8")
    result_file.write(content_name + pprint.pformat(dic))

# 筛选出包含两个以上memberAPI的列表，这样可以方便后续构建API的结构邻接矩阵，我们只需要遍历这些memberAPI的列表，只要两个API同时在某个列表中，就认为有一条边
for i in reversed(range(len(member_api))):
    # 删除了元素以后列表的长度会发生变化，会一直显示list index out of range，reversed函数可以解决这个问题:
    if len(member_api[i][1]) <= 1:
        del member_api[i]

print(structure_tags1_list[0])
print(structure_tags2_list[0])
print(structure_tags3_list[0])
print(member_api)

save_as_py('./Data_mashup_invoke1_20_cate_graph.py',structure_invoke1_list,'mashup_invoke1_20_cate_graph_data=')
save_as_py('./Data_mashup_invoke2_20_cate_graph.py',structure_invoke2_list,'mashup_invoke2_20_cate_graph_data=')
save_as_py('./Data_mashup_tags1_20_cate_graph.py',structure_tags1_list,'mashup_tags1_20_cate_graph_data=')
save_as_py('./Data_mashup_tags2_20_cate_graph.py',structure_tags2_list,'mashup_tags2_20_cate_graph_data=')
save_as_py('./Data_mashup_tags3_20_cate_graph.py',structure_tags3_list,'mashup_tags3_20_cate_graph_data=')
save_as_py('./Data_Pre_memberapi_20_cate.py',member_api,'Pre_memberapi_20_cate_data=')
