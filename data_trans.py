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
from fg_base import dataloader


class Data_to_Picture():
  def __init__(self):
    pass

  def grey_picture(self, data):#灰度图
    result = np.array(data)
    #将长为L的时间序列转成m*n的矩阵， L = m*n
    result = result.reshape((10,-1))
    #矩阵归一化,调用Image
    result = (result - np.min(result)) / (np.max(result) - np.min(result))
    im = Image.fromarray(result*255.0)
    return im.convert('L')
    # im.convert('L').save("1.jpg",format = 'jpeg')

  def recurrent_picture(self, X):#递归图
    rp = RecurrencePlot(threshold='point', percentage=20)
    X_rp = rp.fit_transform(X)
    return X_rp

  def MTF(self, X):#马尔科夫迁变场
    mtf = MarkovTransitionField()
    X_mtf = mtf.fit_transform(X)
    return X_mtf

  def GAF(self, X, methoreds):#格拉米角场
    if methoreds == 'gasf':
      gasf = GramianAngularField(method='summation')
      X_gasf = gasf.fit_transform(X)
      return X_gasf
    if methoreds == 'gadf':
      gadf = GramianAngularField( method='difference')
      X_gadf = gadf.fit_transform(X)
      return X_gadf


  def visualable(self, x,j,i, name='test'):
    plt.figure(dpi=100)
    plt.imshow(x)# 这里要取0，因为只有一个样本
    plt.tight_layout()
    plt.savefig("picture/"+name+"{0}_{1}.png".format(j,i))# save the picture
    plt.show()


  def process_data(self, data):
    data_mean = np.nanmean(data)
    a = pd.DataFrame(data).fillna(data_mean)
    # a.shape # （71，） 这里是把71个数取出来，转换成图
    tmp = a.values
    # tmp = tmp.reshape(1, -1)
    return tmp






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