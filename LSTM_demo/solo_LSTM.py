#!/usr/bin/python3
# -*- encoding: utf-8 -*-

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
import numpy as np
import os
import pandas as pd
import torch
from torch import nn
import datetime
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
DAYS_FOR_TRAIN =1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#如何分配显卡用度


class LSTM_Regression(nn.Module):
    """
        使用LSTM进行回归
        
        参数：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2,days_for_train=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size*days_for_train, output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s,b* h)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x






class Creat_LSTM_data():
    def __init__(self, object_and_label=["date", "label_month_2%", "label_month_15%",
                                   "label_week_7%", "label_week_15%", "code"]):
        self.feature_num = 271
        self.days_for_train = 1
        self.object_and_label = object_and_label
        self.date_and_code = pd.DataFrame()
        self.test_code = []
        self.test_data = []
        pass

        #单只股票原始数据制作
    def data_standard(self,data, lable = "label_month_2%", test_size = 0):
        '''
        data: DataFrame
        '''
        # data_close = pd.read_csv('/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/sh.600015.csv') #读取文件
        if test_size > 0:
            test_num = test_size * data.values.shape[0]
            test_data = data.iloc[-test_num:,:]
            train_data = data.iloc[:data.values.shape[0]-test_num,:]
        #缺失处理
        train_data = data.dropna(axis=0, how='any').copy()
        train_date_and_code = train_data[['date', 'code']]
        nan_columns = self.object_and_label.copy()
        # print(nan_columns)
        nan_columns.remove(lable)
        train_data.drop( columns= nan_columns, inplace=True )
        ss = StandardScaler()
        train_x = ss.fit_transform(train_data.values[:,:-1])
        train_y = train_data.values[:,-1:]
       
        if test_size > 0:
            test_date_and_code = test_data['date', 'code']
            nan_columns = self.object_and_label.remove(lable)
            test_data.drop( columns=nan_columns, inplace=True )
            ss = StandardScaler()
            test_x = ss.fit_transform(test_data.values[:,:-1])
            test_y = test_data.values[:,-1:]
            return train_x, train_y, test_x, test_y  
        else:
            return train_x, train_y, train_date_and_code
   

    def create_multiday_dataset(self, train_x, train_y) -> (np.array, np.array):
        dataset_x, dataset_y= [], []
        for i in range(train_x.shape[0]-self.days_for_train):
            _x = train_x[i:(i+days_for_train),:-1]
            dataset_x.append(_x)
            dataset_y.append(train_y[:,-1:][i+days_for_train-1])
        print('creat_shape', np.array(dataset).shape, np.array(dataset_y).shape)
        return (np.array(dataset_x), np.array(dataset_y))


    ### 训练数据形状
    def feature_reshape(self, train_x, train_y):
        train_x = train_x.reshape(-1,  self.days_for_train , self.feature_num)
        train_y = train_y.reshape(-1, 1)
        # print('train_x',train_x.shape,'  train_y',train_y.shape)
    # 转为pytorch的tensor对象
        train_x = (torch.from_numpy(train_x).to(device)).float()
        train_y = (torch.from_numpy(train_y).to(device)).float()
        return train_x, train_y

# CD = Creat_LSTM_data

# def train_fun(file_list_train,root):
#     model = LSTM_Regression(input_size=CD.feature_num, hidden_size=128, output_size=1, num_layers=2).to(device) # 导入模型并设置模型的参数输入输出层、隐藏层等
#     model_total = sum([param.nelement() for param in model.parameters()]) # 计算模型参数
#     print("Number of model_total parameter: %.8fM" % (model_total/1e6))
#     train_loss = []
#     loss_function = nn.BCELoss().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ### 模型训练
#     for i in range(200):
#         for file in file_list_train:
#             print(file)
#             root_file = os.path.join(root,file)
#             data_close = pd.read_csv(root_file)
#             if data_close.shape[0]<= 600:
#                 continue
#             train_x, train_y, _ = CD.data_standard(data_close)
#             train_x, train_y = CD.feature_reshape(train_x, train_y)
#             t0 = time.time()
#             out = model.forward(train_x)
#             loss = loss_function(out, train_y)
#             loss.requires_grad_(True)
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             train_loss.append(loss.item())
#             print(' Loss:{:.5f}'.format( loss.cpu().item()))
#             # 将训练过程的损失值写入文档保存，并在终端打印出来
#         with open('/home/pc/matrad/leaf/factor/quant_demo/LSTM_demo/log.txt', 'a+') as f:
#             f.write('{} - {}\n'.format(i+1, loss.cpu().item()))
#         if (i+1) % 1 == 0:
#             print('Epoch: {}, Loss:{:.5f}'.format(i+1, loss.cpu().item()))


#     # 画loss曲线
#     plt.figure()
#     plt.plot(train_loss, 'b', label='loss')
#     plt.title("Train_Loss_Curve")
#     plt.ylabel('train_loss')
#     plt.xlabel('epoch_num')
#     plt.savefig('loss.png', format='png', dpi=200)
#     plt.close()
#     torch.save(model.state_dict(), '/home/pc/matrad/leaf/factor/quant_demo/LSTM_demo/model_params.pkl')  # 可以保存模型的参数供未来使用

#     t1=time.time()
#     T=t1-t0
#     print('The training time took %.2f'%(T/60)+' mins.')
#     tt0=time.asctime(time.localtime(t0))
#     tt1=time.asctime(time.localtime(t1))
#     print('The starting time was ',tt0)
#     print('The finishing time was ',tt1)


# # for test
# def model_test(file_list_test, threshold=0.6, evaul_plot=True):
#     model = model.eval() # 转换成测试模式
#     model.load_state_dict(torch.load('model_params.pkl'))  # 读取参数
#     for file in file_list_test:
#         root_file = os.join(root,file)
#         data_close = pd.read_csv(root_file)
#         # if (np.isna(data_close.values).sum()>=data_close.shape[0]).any():
#         #     continue
#         test_x, label, date_and_code = CD.data_standard(data_close)
#         test_x, label = CD.feature_reshape(test_x, label )
#         pred_test = model(test_x.to(torch.float32)) # 全量训练集
#         # print(test_x.shape,pred_test.shape)
#         # print(len(pred_test),len(label))

#         assert len(pred_test) == len(label)
#         pred_prob = ((pred_test.detach().numpy()).reshape((-1,1)))
#         pred_signal = np.sign(np.maximum(pred_test-threshold,0))

#         df = pd.DataFrame()
#         df['label'] = label.squeeze(1)
#         df['pred_prob'] = pred_prob.squeeze(1)
#         df['pred_signal'] = pred_signal.squeeze(1)
#         assert df.values.shape[0] == date_and_code.values.shape[0]
#         df = pd.concat([date_and_code,df])


#         if evaul_plot:
#             label_p = df[df['label']==1].values
#             pred_sig_p = df[df['pred_signal']==1].values
#             accuracy = (label == pred_signal).sum()/len(label)
#             perssion = (label_p == pred_sig_p).sum()/len(pred_sig_p)
#             recall = (label_p == pred_sig_p).sum()/len(label_p)
#             print('accuracy',accuracy, 'perssion',perssion, 'recall',recall)

#             plt.plot(df['pred_signal'].values, 'r', label='prediction')
#             plt.plot(df['label'].values, 'b', label='real')
#             # plt.plot((train_size, test_size), (0, 1), 'g--')  # 分割线 左边是训练数据 右边是测试数据的输出
#             plt.legend(loc='best')
#             plt.savefig('/home/pc/matrad/leaf/factor/quant_demo/'+file+'result.png', format='png', dpi=200)
#             plt.close()

#         df.to_csv('/home/pc/matrad/leaf/factor/strategy/mosel_result/'+file+'mosel_result.png')






# root ='/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data'
# file_list = os.listdir(root)
# train_file_list = file_list[:700]
# test_file_list = file_list[700:]
# train_fun(train_file_list,root)
# model_test(test_file_list)


