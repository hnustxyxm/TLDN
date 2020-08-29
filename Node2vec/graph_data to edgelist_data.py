from Data_API_invoke1_20_cate_graph import API_invoke1_20_cate_graph_data
from Data_API_invoke2_20_cate_graph import API_invoke2_20_cate_graph_data

from Data_mashup_invoke1_20_cate_graph import mashup_invoke1_20_cate_graph_data
from Data_mashup_invoke2_20_cate_graph import mashup_invoke2_20_cate_graph_data

from Data_API_tags1_20_cate_graph import API_tags1_20_cate_graph_data
from Data_API_tags2_20_cate_graph import API_tags2_20_cate_graph_data
from Data_API_tags3_20_cate_graph import API_tags3_20_cate_graph_data

from Data_mashup_tags1_20_cate_graph import mashup_tags1_20_cate_graph_data
from Data_mashup_tags2_20_cate_graph import mashup_tags2_20_cate_graph_data
from Data_mashup_tags3_20_cate_graph import mashup_tags3_20_cate_graph_data

# input_list = [API_invoke1_20_cate_graph_data,API_invoke2_20_cate_graph_data,mashup_invoke1_20_cate_graph_data,mashup_invoke2_20_cate_graph_data,
#               API_tags1_20_cate_graph_data,API_tags2_20_cate_graph_data,API_tags3_20_cate_graph_data,
#               mashup_tags1_20_cate_graph_data,mashup_tags2_20_cate_graph_data,mashup_tags3_20_cate_graph_data]
#
# output_list = ['Data_API_invoke1_20_cate_undirected.edgelist','Data_API_invoke2_20_cate_undirected.edgelist','Data_mashup_invoke1_20_cate_undirected.edgelist','Data_mashup_invoke2_20_cate_undirected.edgelist',
#                'Data_API_tags1_20_cate_undirected.edgelist','Data_API_tags2_20_cate_undirected.edgelist','Data_API_tags3_20_cate_undirected.edgelist',
#                'Data_mashup_tags1_20_cate_undirected.edgelist','Data_mashup_tags2_20_cate_undirected.edgelist','Data_mashup_tags3_20_cate_undirected.edgelist']

input_list = [API_invoke1_20_cate_graph_data,API_invoke2_20_cate_graph_data]

output_list = ['Data_API_invoke1_20_cate_undirected.edgelist','Data_API_invoke2_20_cate_undirected.edgelist']

# input_list = {1:API_invoke1_20_cate_graph_data}

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

# test_list_data = pd.DataFrame(test_list)
# test_list_data.to_csv('./test.edgelist', encoding='utf-8', index=0)


