#!/usr/bin/python3
# -*- encoding: utf-8 -*-

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
import numpy as np

import pandas as pd
import torch
from torch import nn
import datetime
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#如何分配显卡用度
DAYS_FOR_TRAIN = 5


import numpy as np

import pandas as pd
import torch
from torch import nn
import datetime
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#如何分配显卡用度
DAYS_FOR_TRAIN = 1


class LSTM_Regression(nn.Module):
    """
        使用LSTM进行回归
        
        参数：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size*DAYS_FOR_TRAIN, output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, _x):
        # print('_x.shape',_x.shape)
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        # print('x.shape',x.shape)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s,b* h)
        # print('x.shape',x.shape)
        x = self.fc(x)
        # print('x_fc.shape',x.shape)
        # x = x.view(s, b, -1)  # 把形状改回来
        # print('x_final.shape',x.shape)
        x = self.sigmoid(x)
        # print('x_final.shape',x.shape)
        return x
        #x.shape torch.Size([2547, 548, 128])
        # x.shape torch.Size([1395756, 128])
        # x_final.shape torch.Size([1395756, 1])问题：3维与2维。





class Creat_data():
    def __init__(self, object_and_label=["date", "label_month_%2", "label_month_15%", "adjustflag", "tradestatus",
                                   "label_week_7%", "label_week_15%", "code"]):
        self.feature_num = 548
        self.days_for_train = 5
        self.test_code = []
        self.test_data = []
        pass

    def data_standard(self,):
        data_close = pd.read_csv('/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/sh.600015.csv') #读取文件
        data_close.drop( columns=['date', "label_week_15%", "adjustflag","tradestatus","label_week_7%",  "code"], inplace=True )
        data_close["label_month_15%" ] = np.sign(np.maximum((data_close["lagRet"].rolling(22).sum()).shift(-22) - np.log(1.03), 0) )
        df_train = data_close.dropna(axis=0, how='any').copy()

        data_close = df_train.values
        data_close_shape = data_close.shape
        ss = StandardScaler()
        data_close_ss = ss.fit_transform(data_close)
        test_x, dataset_y = data_close_ss[:,:-1], data_close[:,-1:]
  
    # 划分训练集和测试集，70%作为训练集
    def solo_stock_split(self,train_size):
        train_size = int(len(test_x) * train_size)
        train_x = test_x[:train_size]
        train_y = dataset_y[:train_size]
        print('before reshape train_x',train_x.shape,'  train_y',train_y.shape)
    

    def create_multiday_dataset(self, data) -> (np.array, np.array):
        test_x, dataset_y= [], []
        for i in range(data.shape[0]-self.days_for_train):
            _x = data[i:(i+days_for_train),:-1]
            test_x.append(_x)
            # print('data[:-1][i+days_for_train]',data[:-1][i+days_for_train])
            dataset_y.append(data[:,-1:][i+days_for_train-1])
        print('creat_shape', np.array(test_x).shape, np.array(dataset_y).shape)
        return (np.array(test_x), np.array(dataset_y))


    ### 训练数据形状
    def feature_reshape(self,):
        train_x = train_x.reshape(-1,  DAYS_FOR_TRAIN , self.feature_num)
        train_y = train_y.reshape(-1, 1)
        print('train_x',train_x.shape,'  train_y',train_y.shape)
    # 转为pytorch的tensor对象
        train_x = torch.from_numpy(train_x).to(device)
        train_y = torch.from_numpy(train_y).to(device)

CD = Creat_data()

def train(train_x, train_y):
    model = LSTM_Regression(input_size=CD.feature_num, hidden_size=128, output_size=1, num_layers=2).to(device) # 导入模型并设置模型的参数输入输出层、隐藏层等
    model_total = sum([param.nelement() for param in model.parameters()]) # 计算模型参数
    print("Number of model_total parameter: %.8fM" % (model_total/1e6))
    train_loss = []
    loss_function = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

### 模型训练
    for i in range(200):
        t0 = time.time()
        out = model(train_x.float())
        loss = loss_function(out.float(), train_y.float())
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss.append(loss.item())
        
        # 将训练过程的损失值写入文档保存，并在终端打印出来
        with open('log.txt', 'a+') as f:
            f.write('{} - {}\n'.format(i+1, loss.cpu().item()))
        if (i+1) % 1 == 0:
            print('Epoch: {}, Loss:{:.5f}'.format(i+1, loss.cpu().item()))


    # 画loss曲线
    plt.figure()
    plt.plot(train_loss, 'b', label='loss')
    plt.title("Train_Loss_Curve")
    plt.ylabel('train_loss')
    plt.xlabel('epoch_num')
    plt.savefig('loss.png', format='png', dpi=200)
    plt.close()
    torch.save(model.state_dict(), 'model_params.pkl')  # 可以保存模型的参数供未来使用

    t1=time.time()
    T=t1-t0
    print('The training time took %.2f'%(T/60)+' mins.')
    tt0=time.asctime(time.localtime(t0))
    tt1=time.asctime(time.localtime(t1))
    print('The starting time was ',tt0)
    print('The finishing time was ',tt1)


# for test
def model_test(test_x, label, threshold=0.6):
    model = model.eval() # 转换成测试模式
    model.load_state_dict(torch.load('model_params.pkl'))  # 读取参数

    # 注意这里用的是全集 模型的输出长度会比原数据少DAYS_FOR_TRAIN 填充使长度相等再作图
    test_x = test_x.reshape(-1,  DAYS_FOR_TRAIN , 548)  # (seq_size, batch_size, feature_size)
    test_x = torch.tensor(test_x)

    pred_test = model(test_x.to(torch.float32)) # 全量训练集
    print(test_x.shape,pred_test.shape)
    print(len(pred_test),len(label))

    assert len(pred_test) == len(label)
    pred_prob = ((pred_test.detach().numpy()).reshape((-1,1)))
    pred_signal = np.sign(np.maximum(pred_test-threshold,0))

    df = pd.DataFrame()
    df['label'] = label.squeeze(1)
    df['pred_prob'] = pred_prob.squeeze(1)
    df['pred_signal'] = pred_signal.squeeze(1)
    return df


def evaul_plot():
    label_p = df[df['label']==1].values
    pred_sig_p = df[df['pred_signal']==1].values
    accuracy = (label == pred_signal).sum()/len(label)
    perssion = (label_p == pred_sig_p).sum()/len(pred_sig_p)
    recall = (label_p == pred_sig_p).sum()/len(label_p)
    print('accuracy',accuracy, 'perssion',perssion, 'recall',recall)

    plt.plot(df['pred_signal'].values, 'r', label='prediction')
    plt.plot(df['label'].values, 'b', label='real')
    # plt.plot((train_size, test_size), (0, 1), 'g--')  # 分割线 左边是训练数据 右边是测试数据的输出
    plt.legend(loc='best')
    plt.savefig('result.png', format='png', dpi=200)
    plt.close()
