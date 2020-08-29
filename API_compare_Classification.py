import torch
from Data_API_X_train_01234 import API_X_train_01234_data
from Data_API_X_test_01234 import API_X_test_01234_data
from Data_API_Y_train_01234 import API_Y_train_01234_data
from Data_API_Y_test_01234 import API_Y_test_01234_data

from Data_API_X_train_1234 import API_X_train_1234_data
from Data_API_X_test_1234 import API_X_test_1234_data
from Data_API_Y_train_1234 import API_Y_train_1234_data
from Data_API_Y_test_1234 import API_Y_test_1234_data

from Data_API_X_train_0234 import API_X_train_0234_data
from Data_API_X_test_0234 import API_X_test_0234_data
from Data_API_Y_train_0234 import API_Y_train_0234_data
from Data_API_Y_test_0234 import API_Y_test_0234_data

from Data_API_X_train_0134 import API_X_train_0134_data
from Data_API_X_test_0134 import API_X_test_0134_data
from Data_API_Y_train_0134 import API_Y_train_0134_data
from Data_API_Y_test_0134 import API_Y_test_0134_data

from Data_API_X_train_0124 import API_X_train_0124_data
from Data_API_X_test_0124 import API_X_test_0124_data
from Data_API_Y_train_0124 import API_Y_train_0124_data
from Data_API_Y_test_0124 import API_Y_test_0124_data

from Data_API_X_train_0123 import API_X_train_0123_data
from Data_API_X_test_0123 import API_X_test_0123_data
from Data_API_Y_train_0123 import API_Y_train_0123_data
from Data_API_Y_test_0123 import API_Y_test_0123_data

from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim
import pandas as pd

# 注意：0代表LDA表征向量，1代表Doc2vec表征向量，2代表TFIDF表征向量，3代表服务调用结构表征向量，4代表服务标签结构表征向量。
# 缺少的数字就表示对应缺少的表征向量

# 单独训练
# x = LDA_API_20_cate_data
# x = torch.FloatTensor(x)

# x = LDA_API_20_cate_data_filter
# x = torch.FloatTensor(x)

# 联合训练
# x0 = LDA_API_20_cate_data
# x1 = Doc2API_20_cate_data
# x2 = TFIDF_API_20_cate_data
# x3 = Node2_API_invoke1_20_cate_data
# x4 = Node2_API_tags1_20_cate_data
#
# x0 = torch.FloatTensor(x0)
# x1 = torch.FloatTensor(x1)
# x2 = torch.FloatTensor(x2)
# x3 = torch.FloatTensor(x3)
# x4 = torch.FloatTensor(x4)
#
# x = torch.cat((x0, x1, x2, x3, x4), 1)
# 按维数0拼接（竖着拼）; 按维数1拼接（横着拼）


def roc_drawing(out,labels,tif_path,tif_Name):
    num_classes = 20
    scores = torch.softmax(out, dim=1).detach().numpy()  # out = model(data)
    binary_label = label_binarize(labels, classes=list(range(num_classes)))  # num_classes=10

    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(binary_label[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(binary_label.ravel(), scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # 画图
    plt.figure(figsize=(8, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    for i in range(20):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(tif_Name)
    plt.legend(loc="lower right")
    plt.savefig(tif_path, bbox_inches='tight')
    # plt.show()

    return roc_auc["micro"],roc_auc["macro"]


class Net(nn.Module):
    def __init__(self,n_feature,n_hidden,n_hidden1,n_hidden2,n_hidden3,hidden_layer,n_out):
        super(Net,self).__init__()
        self.hidden = nn.Linear(n_feature,n_hidden)
        self.hidden1 = nn.Linear(n_hidden,n_hidden1)
        self.hidden2 = nn.Linear(n_hidden1,n_hidden2)
        self.hidden3 = nn.Linear(n_hidden2,n_hidden3)
        # self.hidden4 = nn.Linear(n_hidden3,n_hidden4)
        if hidden_layer == 1:
            self.out = nn.Linear(n_hidden,n_out)
        if hidden_layer == 2:
            self.out = nn.Linear(n_hidden1,n_out)
        if hidden_layer == 3:
            self.out = nn.Linear(n_hidden2,n_out)
        if hidden_layer == 4:
            self.out = nn.Linear(n_hidden3,n_out)
        # if hidden_layer == 5:
        #     self.out = nn.Linear(n_hidden4,n_out)

    def forward(self, x, hidden_layer):
        # relu的效果比sigmoid要好
        if hidden_layer == 1:
            x = F.relu(self.hidden(x))
        if hidden_layer == 2:
            x = F.relu(self.hidden(x))
            x = F.relu(self.hidden1(x))
        if hidden_layer == 3:
            x = F.relu(self.hidden(x))
            x = F.relu(self.hidden1(x))
            x = F.relu(self.hidden2(x))
        if hidden_layer == 4:
            x = F.relu(self.hidden(x))
            x = F.relu(self.hidden1(x))
            x = F.relu(self.hidden2(x))
            x = F.relu(self.hidden3(x))
        # if hidden_layer == 5:
        #     x = F.relu(self.hidden(x))
        #     x = F.relu(self.hidden1(x))
        #     x = F.relu(self.hidden2(x))
        #     x = F.relu(self.hidden3(x))
        #     x = F.relu(self.hidden4(x))
        x = self.out(x)
        out = F.log_softmax(x,dim=1)
        return out


input_list_train_X = [API_X_train_01234_data,API_X_train_1234_data,API_X_train_0234_data,
                      API_X_train_0134_data,API_X_train_0124_data,API_X_train_0123_data]
input_list_test_X = [API_X_test_01234_data,API_X_test_1234_data,API_X_test_0234_data,
                     API_X_test_0134_data,API_X_test_0124_data,API_X_test_0123_data]
input_list_train_Y = [API_Y_train_01234_data,API_Y_train_1234_data,API_Y_train_0234_data,
                      API_Y_train_0134_data,API_Y_train_0124_data,API_Y_train_0123_data]
input_list_test_Y = [API_Y_test_01234_data,API_Y_test_1234_data,API_Y_test_0234_data,
                     API_Y_test_0134_data,API_Y_test_0124_data,API_Y_test_0123_data]

# output_list_jpg_path = ['./Fin_result_data/API_01234_roc.jpg','./Fin_result_data/API_1234_roc.jpg','./Fin_result_data/API_0234_roc.jpg',
#                         './Fin_result_data/API_0134_roc.jpg','./Fin_result_data/API_0124_roc.jpg','./Fin_result_data/API_0123_roc.jpg']

output_list_tif_path = ['./Fin_result_data/API_TLDN_roc.tif','./Fin_result_data/API_TD-IT_roc.tif','./Fin_result_data/API_TL-IT_roc.tif',
                        './Fin_result_data/API_LD-IT_roc.tif','./Fin_result_data/API_TLD-T_roc.tif','./Fin_result_data/API_TLD-I_roc.tif']

output_list_tif_Name = ['TLDN ROC in API dataset','TD-IT ROC in API dataset','TL-IT ROC in API dataset',
                        'LD-IT ROC in API dataset','TLD-T ROC in API dataset','TLD-I ROC in API dataset',]

output_list_compare_method = [[0],[1],[2],[3],[4],[5]]

loss_result = []
mi_f1_result = []
ma_f1_result = []
mi_auc_result = []
ma_auc_result = []

for k in range(6):
    x_train = input_list_train_X[k]
    x_test = input_list_test_X[k]
    y_train = input_list_train_Y[k]
    y_test = input_list_test_Y[k]
    input_dim = len(x_train[0])

    x_train = torch.FloatTensor(x_train)
    y_train = torch.LongTensor(y_train)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)

    x_train = Variable(x_train)
    y_train = Variable(y_train)
    x_test = Variable(x_test)
    y_test = Variable(y_test)

    min_loss = 10
    parameter_list = [[2,0.15,12500]]  # 将最佳参数代入即可，根据结果分析，考虑到复杂度等问题可得，API的最佳参数为[2,0.15,12500]

    for i in range(len(parameter_list)):
        # n_feature表示输入特征向量的维度，n_hidden表示隐藏层的单元数，n_out表示输出的类别数
        net = Net(n_feature=input_dim,n_hidden=256,n_hidden1=128,n_hidden2=64,n_hidden3=32,hidden_layer=parameter_list[i][0],n_out=20)

        optimizer = torch.optim.SGD(net.parameters(),lr=parameter_list[i][1])

        epochs = parameter_list[i][2]

        for j in range(epochs):
            predict_train = net(x_train,hidden_layer=parameter_list[i][0])
            loss = F.nll_loss(predict_train,y_train)  # 输出层 用了log_softmax 则需要用这个误差函数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.set_printoptions(profile="full")
            if j % 1000 == 0:
                print('第'+str(j)+'次epoch遍历')
        #   if i == epochs-1:
        #       torch.save(net,'net.pkl')
        #
        # net1 = torch.load('net.pkl')

        # 进行测试
        predict_test = net(x_test,hidden_layer=parameter_list[i][0])
        loss = F.nll_loss(predict_test,y_test)
        _, pred = torch.max(predict_test, 1)
        # torch.max(input, dim) 函数
        # output = torch.max(input, dim)
        # 输入
        # input是softmax函数输出的一个tensor
        # dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
        # 输出
        # 函数会返回两个tensor，第一个tensor是每行的最大值，softmax的输出中最大的是1，所以第一个tensor是全1的tensor；第二个tensor是每行最大值的索引。

        ma_f1 = f1_score(y_test.detach().numpy(), pred.detach().numpy(), average='macro')
        mi_f1 = f1_score(y_test.detach().numpy(), pred.detach().numpy(), average='micro')
        # 将tensor转化为数组时，待转换类型的PyTorch Tensor变量带有梯度，直接将其转换为numpy数据将破坏计算图，因此numpy拒绝进行数据转换，实际上这是对开发者的一种提醒。如果自己在转换数据时不需要保留梯度信息，可以在变量转换之前添加detach()调用。假设原来的写法是：
        # aaa.cpu().numpy()
        # 那么现在改为
        # aaa.cpu().detach().numpy()即可。
        roc_auc_micro,roc_auc_macro = roc_drawing(predict_test,y_test.detach().numpy(),output_list_tif_path[k],output_list_tif_Name[k])
        # 注意，这里传入的是tensor向量predict，不是数组向量pred。原因是因为源码要求传入的是tensor

        loss_result.append(loss.item())
        mi_f1_result.append(mi_f1.item())
        ma_f1_result.append(ma_f1.item())
        mi_auc_result.append(roc_auc_micro)
        ma_auc_result.append(roc_auc_macro)

        if min_loss >= loss:
            min_loss = loss

        print('第'+str(i)+'次遍历')
        print(str(parameter_list[0])+'min_loss:'+str(min_loss))

        # for i in range(1):
        #     print(str(i)+"loss:"+str(loss.item()))
        #     print(str(i)+"ma_f1:"+str(ma_f1.item()))
        #     print(str(i)+"mi_f1:"+str(mi_f1.item()))
        #     roc_auc_micro,roc_auc_macro = roc_drawing(predict_test,y_test.detach().numpy())
        #     # 注意，这里传入的是tensor向量predict，不是数组向量pred。原因是因为源码要求传入的是tensor
        #     print(roc_auc_micro,'\n',roc_auc_macro)

# fin_data = zip(output_list_compare_method,loss_result,mi_f1_result,ma_f1_result,mi_auc_result,ma_auc_result)
#
# API_result_data = pd.DataFrame(list(fin_data),columns=['compare_method','loss_result','mi_f1_result','ma_f1_result',
#                                                        'mi_auc_result','ma_auc_result'])
# API_result_data.to_csv('./Fin_result_data/API_compare_result_data.csv',encoding='utf-8',index=0)


print('loss_min:',min(loss_result))
a = loss_result.index(min(loss_result))
print('loss_min_index:',a)
print('parameter_list:',output_list_compare_method[a])

print('mi_f1_max:',max(mi_f1_result))
a = mi_f1_result.index(max(mi_f1_result))
print('mi_f1_max_index:',a)
print('parameter_list:',output_list_compare_method[a])

print('ma_f1_max:',max(ma_f1_result))
a = ma_f1_result.index(max(ma_f1_result))
print('ma_f1_max_index:',a)
print('parameter_list:',output_list_compare_method[a])

print('mi_auc_max:',max(mi_auc_result))
a = mi_auc_result.index(max(mi_auc_result))
print('mi_auc_max_index:',a)
print('parameter_list:',output_list_compare_method[a])

print('ma_auc_max:',max(ma_auc_result))
a = ma_auc_result.index(max(ma_auc_result))
print('ma_auc_max_index:',a)
print('parameter_list:',output_list_compare_method[a])


