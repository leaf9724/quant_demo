from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from pyts.image import RecurrencePlot
from pyts.image import MarkovTransitionField
import pickle
import pandas as pd
import numpy as np
from PIL import Image
import sys
sys.path.append('/home/pc/matrad/leaf/factor/quant_demo')
from LSTM_demo.solo_LSTM import Creat_LSTM_data


class Data_to_Picture(Creat_LSTM_data):


  def grey_picture(self, data,save_dic, root ='/home/pc/matrad/leaf/factor/daily_data/data_processed/grey_vaild'):#灰度图
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    im = Image.fromarray(data*255.0)
    im.convert('L').save(root+"/grey{0}_{1}_{2}_{3}.jpg".format(save_dic['name'],save_dic['inx'],save_dic['date_code'],save_dic['label']),format = 'jpeg')

    # im.convert('L').save("1.jpg",format = 'jpeg')

  def recurrent_picture(self, X,root ='/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/picture'):#递归图
    rp = RecurrencePlot(threshold='point', percentage=20)
    X_rp = rp.fit_transform(X)
    plt.figure(figsize=(5, 5))
    plt.imshow(X_rp[0], cmap='binary', origin='lower')
    plt.tight_layout()
    plt.savefig(root+"/rp{0}_{1}_{2}_{3}.jpg".format(j,i))

    return X_rp

  def MTF(self, X,root ='/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/picture'):#马尔科夫迁变场
    mtf = MarkovTransitionField()
    X_mtf = mtf.fit_transform(X)
    plt.figure(figsize=(5, 5))
    plt.imshow(X_mtf[0], cmap='rainbow', origin='lower')
    plt.colorbar(fraction=0.0457, pad=0.04)
    plt.tight_layout()
    plt.savefig(root+"/mtf.jpg")


  def GAF(self, X, methoreds='gasf',root ='/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/picture'):#格拉米角场
    if methoreds == 'gasf':
      gasf = GramianAngularField(method='summation')
      picture = gasf.fit_transform(X)

    if methoreds == 'gadf':
      gadf = GramianAngularField( method='difference')
      picture = gadf.fit_transform(X)

    plt.imshow(picture[0], cmap='rainbow', origin='lower')
    plt.tight_layout()
    plt.savefig(root+"/gaf.jpg")



  def visualable(self, x, name='test'):
    plt.figure(dpi=100)
    plt.imshow(x)# 这里要取0，因为只有一个样本
    plt.tight_layout()
    plt.savefig("picture.png")
    # plt.savefig("picture/"+name+"{0}_{1}.png".format(j,i))# save the picture
    plt.show()


  def grey_process(self,dimen = 1):
    data = pd.read_csv('/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/sh.600036.csv')
    if dimen == 1:
      train_x, train_y, date_and_code = super().data_standard(data)
      shape = train_x.shape    
      for i in range(shape[0]):
        if (np.sqrt(shape[1]) % 1) == 0 :
          temp = temp.reshape(int(np.sqrt(shape[1])),int(np.sqrt(shape[1]))+1)
        try:
          temp = temp.reshape(int(np.sqrt(shape[1])),int(np.sqrt(shape[1]))+1)
        except:
          temp = np.append(train_x[i,:], 0)
          temp = temp.reshape(int(np.sqrt(shape[1])),int(np.sqrt(shape[1]))+1)
        dic_save={'name':'sh.600031','inx':str(i),'label':str(train_y[i]),'date_code':str(date_and_code.values[i,:])}
        self.grey_picture(temp,dic_save)

    if dimen != 1:
      train_x, train_y, date_and_code = super().data_standard(data)
      train_x, train_y = super().create_multiday_dataset(train_x, train_y)
      shape = train_x.shape    
      for i in range(shape[0]):
        temp = train_x[i,:,:]
        self.grey_picture(temp,name= 'sh.600031',inx=str(i))




  # def process_data(self,):
  #   data_close = pd.read_csv('/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/sh.600031.csv')
  #   train_x, train_y, date_and_code = super().data_standard(data_close)
  #   print(train_x[500,:].shape)
  #   train_x0 = np.append(train_x[500,:], 0)
  #   train_x0 = train_x0.reshape(16,17)
  #   self.grey_picture(train_x0)
  #   self.recurrent_picture(train_x[495:500,:])
  #   self.MTF(train_x[495:500,:])
  #   self.GAF(train_x[495:500,:])
  #   # self.visualable(grey1)
  #   print(train_x0.shape)

D2P = Data_to_Picture()
D2P.grey_process()






class Data_to_Graph():
    def __init__(self):
        pass

    def data_to_graph(self, data_x, data_y):#单行数据生成单个graph,每个graph有一个标签
        series = ...
        graphEdges = visibility_graph(series)
        list1 = []
        list2 = []
        print('list1', list1)
        for j in graphEdges:
            list1.append(j[0])
            list2.append(j[1])
            edges_index = torch.tensor([list1, list2])
            x = np.expand_dims(series.values, 1).tolist()
            graph_data = Data(x=x, edge_index=edges_index, y=y)
            graph_list.append(graph_data)
        inde = list(k for k in range(318))
        graph_list.train_idx = torch.tensor(random.sample(inde, 260), dtype=torch.long)
        graph_list.test_mask = torch.tensor(random.sample(inde, 56), dtype=torch.long)
        np.savez("picture/graph", graph_list)


    def visibility_graph(self, series):
        visibility_graph_edges = []
        # convert list of magnitudes into list of tuples that hold the index
        tseries = []
        n = 1
        for magnitude in series:
            tseries.append((n, magnitude))
            n += 1

        for a, b in combinations(tseries, 2):
            # two points, maybe connect
            # 实现（1,2)（1,3）(1,4)(1,5)--(2,3)(2,4)(2,5)--(3,4)(3,5)--(4,5)任意两个边相互比较
            (ta, ya) = a
            (tb, yb) = b
            connect = True
            # 此处需要多留意，ta是1到k，而tseris是从0下标开始  所以此处不能不是[ta+1:tb]
            medium = tseries[ta:tb - 1]

            for tc, yc in medium:
                # print yc,(yb + (ya - yb) * ( float(tb - tc) / (tb - ta) ))#一定要float(tb-tc)/(tb-ta)因为计算机里1/2为0,1.0/2才为0.5
                if yc > yb + (ya - yb) * (float(tb - tc) / (tb - ta)):
                    connect = False
            if connect:
                visibility_graph_edges.append((ta, tb))
        return visibility_graph_edges