import os
import numpy as np
import pandas as pd
import talib as ta
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm


class Feature_engine():
    def __init__(self):
        self.data_all = pd.DataFrame()
        self.data_train = pd.DataFrame()
        self.data_test = pd.DataFrame()
        self.row_path="/home/pc/matrad/leaf/factor/daily_data/price_data"
        self.processed_path="/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/"
        self.feature_path="/home/pc/matrad/leaf/factor/daily_data"
        self.file_list = os.listdir(self.row_path)#[:50]
        self.label = ' '

    def price_adjust(self, data):
        data["close"] = data["close"] * data["adjustflag"]
        data["open"] = data["open"] * data["adjustflag"]
        data["high"] = data["high"] * data["adjustflag"]
        data["low"] = data["low"] * data["adjustflag"]
        data["preclose"] = data["preclose"] * data["adjustflag"]
        data['close_t-1'] = data["close"].shift(1)
        data["lagRet"] = np.log(data["close"] / data["close_t-1"])

        for i in range(10, 500, 20):
            data["lagRet_" + str(i)] = data["lagRet"].rolling(i).sum()

        data.drop(columns=['close_t-1'], inplace=True)
        return data

    def daily_signal(self, data):
        for j in range(3, 10, 2):
            data["ret_to_route_" + str(j)] = data["lagRet"].rolling(j).sum()

        for k in range(10, 50, 5):
            data["ret_to_route_" + str(k)] = data["lagRet"].rolling(k).sum()

        for i in range(5, 30, 5):
            data['roll_min'] = data["close"].rolling(i).min()
            data['roll_max'] = data["close"].rolling(i).max()
            data["rpos" + str(i)] = (data["close"] - data['roll_min'] ) / (data['roll_max'] - data['roll_min'] )
            data.drop(columns=['roll_min','roll_max'], inplace=True)

        for i in range(30, 120, 15):
            data['roll_min'] = data["close"].rolling(i).min()
            data['roll_max'] = data["close"].rolling(i).max()
            data["rpos" + str(i)] = (data["close"] - data['roll_min'] ) / (data['roll_max'] - data['roll_min'] )
            data.drop(columns=['roll_min','roll_max'], inplace=True)

        return data

    def talib_overlap(self, data):
        data['BBAND_upperband'], data['BBAND_middleband'], data['BBAND_lowerband'] = ta.BBANDS(data['close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    
        for i in range(10,201,10):
            data['DEMA'+str(i)] = ta.DEMA(data['close'], timeperiod=30)
            data['EMA'+str(i)] = ta.EMA(data['close'], timeperiod=30)
            data['TEMA'+str(i)] = ta.TEMA(data['close'], timeperiod=30)
            data['TRIMA'+str(i)] = ta.TRIMA(data['close'], timeperiod=30)
            data['WMA'+str(i)] = ta.WMA(data['close'], timeperiod=30)
            data['KAMA'+str(i)] = ta.KAMA(data['close'], timeperiod=30)
            data['SMA'+str(i)] = ta.SMA(data['close'], timeperiod=30)
            data['MA'+str(i)] = ta.MA(data['close'], timeperiod=30, matype=0)
        for i in range(14,211, 14):         
            data['MIDPOINT'] = ta.MIDPOINT(data['close'], timeperiod=14)
            data['MIDPRICE'] = ta.MIDPRICE(data['high'], data['low'], timeperiod=14)

        # data['MAMA_mama'],data['MAMA_fama'] = ta.MAMA(data['close'], fastlimit=1, slowlimit=1)
        # data['MAVP'] = ta.MAVP(data['close'], periods, minperiod=2, maxperiod=30, matype=0)
        data['SAR'] = ta.SAR(data['high'], data['low'], acceleration=0, maximum=0)
        data['SAREXT'] = ta.SAREXT(data['high'], data['low'], startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
        data['T3'] = ta.T3(data['close'], timeperiod=5, vfactor=0)
        data['HT_TRENDLINE'] = ta.HT_TRENDLINE(data['close'])
        return data

    def talib_volume(self, data):
        data['AD'] = ta.AD(data['high'], data['low'], data['close'], data['volume'])
        data['ADSC'] = ta.ADOSC(data['high'], data['low'], data['close'], data['volume'], fastperiod=3, slowperiod=10)
        data['OBV'] = ta.OBV(data['close'], data['volume'])
        return data  

    def talib_volatity(self, data):
        for i in range(14,211, 14):
            data['ATR'+str(i)] = ta.ATR(data['high'], data['low'], data['close'], timeperiod=14)
            data['NATR'+str(i)] = ta.NATR(data['high'], data['low'], data['close'], timeperiod=14)

        data['TRANGE'] = ta.TRANGE(data['high'], data['low'], data['close'])
        return data

    def talib_cycle(self, data):
        data['HT_DCPERIOD'] = ta.HT_DCPERIOD(data['close'])
        data['HT_DCPHASE'] = ta.HT_DCPHASE(data['close'])
        data['HT_PHASOR_inphase'], data['HT_PHASOR_quadrature'] = ta.HT_PHASOR(data['close'])
        data['HT_SINE_sine'], data['HT_SINE_leadsine'] = ta.HT_SINE(data['close'])
        data['HT_TRENDMODE'] = ta.HT_TRENDMODE(data['close'])
        return data

    def data_standard(self, data):
        data_obj = data.select_dtypes(exclude='number').copy()
        data = data.select_dtypes(include='number').copy()
        columns = np.array(data.columns)
        columns_obj = np.array(data_obj.columns)
        data_columns =np.concatenate([columns_obj,columns],axis = 0)

        data = np.divide(data.values - np.nanmean(data.values, axis=0), np.nanstd(data.values, axis=0))
        data = np.concatenate([data_obj.values,data], axis=1)
        data = pd.DataFrame(data)
        data.columns = data_columns
        return data

    def label_make(self, data):
        data["label_month_15%" ] = np.sign(np.maximum((data["lagRet"].rolling(20).sum()).shift(-20) - np.log(1.15), 0) )
        data["label_week_7%" ] = np.sign(np.maximum((data["lagRet"].rolling(5).sum()).shift(-5) - np.log(1.07), 0) )
        data["label_week_15%" ] = np.sign(np.maximum((data["lagRet"].rolling(5).sum()).shift(-5) - np.log(1.15), 0) )
        data["label_month_%2" ] = np.sign(np.maximum((data["lagRet"].rolling(20).sum()).shift(-20) - np.log(1.02), 0) )
        return data

    def forward(self, target='renew'):
        for file in tqdm(self.file_list):  # os.listdir('.')遍历文件夹内的每个文件名，并返回一个包含文件名的list
            # print(file)
            data_org = pd.read_csv(self.row_path + "/" + file)
            if target == 'renew':
                data = data.iloc[-600:,:]
            else:
                data =data_org.iloc
                           
            data = self.price_adjust(data)
            data = self.daily_signal(data)
            data = self.talib_overlap(data)
            data = self.talib_volatity(data)
            data = self.talib_volume(data)
            data = self.talib_cycle(data)
            data = self.data_standard(data)
            data = self.label_make(data)

            if target == 'renew':
                data = pd.concat([data_org,data.iloc[-1,:]],axis=0)
            data.to_csv(self.processed_path + "/" + file, index=False)
            self.data_all=pd.concat([data,self.data_all],axis=0)
            self.data_train= pd.concat([data.iloc[:-150,:],self.data_train],axis=0)
            self.data_test = pd.concat([data.iloc[-150:,:],self.data_test],axis=0)

        self.data_all = data_all[(data_all.isST != 1)]  # &(dt.c1!=56)]
        self.data_all.drop(columns=["isST",'preclose'], inplace=True)
        self.data_all.to_csv(self.feature_path + "/" + "feature_dall.csv", index=False)
        self.data_train.to_csv(self.feature_path + "/" + "feature_dtrain.csv", index=False)
        self.data_test.to_csv(self.feature_path + "/" + "feature_dtest.csv", index=False)


            
file_manuer = Feature_engine()
file_manuer.forward('all')



# def signal_calc(data):
#     #bolling
#     for i in [5,10,20,30,60,90,120,150]:
#         data['bbl_upperband'+str(i)], data['bbl_middleband'], data['bbl_lowerband'] = ta.BBANDS(data['close'], timeperiod=i, nbdevup=2, nbdevdn=2, matype=0)

#     for i in range(7,71,7):
#         data['ADX'+str(i)] = ta.ADX(data['high'], data['low'], data['close'], timeperiod=i)

#     for i in range(77,260,21):
#         data['ADX'+str(i)] = ta.ADX(data['high'], data['low'], data['close'], timeperiod=i)        

#     for i in range(1,11):
#         data['APO'+str(i)] = ta.APO(data['close'], fastperiod=12*i, slowperiod=26*i, matype=0)


#     for i in range(1,11):
#         data['aroondown'+str(i)], data['aroonup'+str(i)] = ta.AROON(data['high'], data['low'], timeperiod=14*i)

#         data['AROONOSC'+str(i)] = ta.AROONOSC(data['high'], data['low'], timeperiod=14*i)

#         data['CCI'+str(i)] = ta.CCI(data['high'], data['low'], data['close'], timeperiod=14*i)

#         data['CMO'+str(i)] = ta.CMO(data['close'], timeperiod=14*i)

#         data['DX'+str(i)] = ta.DX(data['high'], data['low'], data['close'], timeperiod=14*i)

#         data['MIF'+str(i)] = ta.MFI(data['high'], data['low'], data['close'], data['volume'], timeperiod=14*i)

#         data['MINUS_DI'+str(i)] = ta.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=14*i)

#         data['MINUS_DM'+str(i)] = ta.MINUS_DM(data['high'], data['low'], timeperiod=14*i)
#         data['PLUS_DI'+str(i)] = ta.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=14*i)
#         data['PLUS_DM'+str(i)] = ta.PLUS_DM(data['high'], data['low'], timeperiod=14*i)
#         data['RSI'+str(i)] = ta.RSI(data['close'], timeperiod=14*i)
#         data['WILLR'+str(i)] = ta.WILLR(data['high'], data['low'], data['close'], timeperiod=14*i)
#         data['ADXR'+str(i)] = ta.ADXR(data['high'], data['low'], data['close'], timeperiod=14*i)


#     data['BOP'] = ta.BOP(data['open'], data['high'], data['low'], data['close'])

#     data['macd'],data['macdsignal'], data['macdhist'] = ta.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)

#     data['macd_ext'],data['macdsignal_ext'], data['macdhist_ext'] = ta.MACDEXT(data['close'], fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)

#     data['macd_fix'],data['macdsignal_fix'], data['macdhist_fix'] = ta.MACDFIX(data['close'], signalperiod=9)

#     data['PPO'] = ta.PPO(data['close'], fastperiod=12, slowperiod=26, matype=0)

#     for i in range(10,201,10):
#         data['ROC'+str(i)] = ta.ROC(data['close'], timeperiod=10)

#         data['ROCP'+str(i)] = ta.ROCP(data['close'], timeperiod=10)

#         data['ROCR'+str(i)] = ta.ROCR(data['close'], timeperiod=10)

#         data['ROCR100'+str(i)] = ta.ROCR100(data['close'], timeperiod=10)



#     data['slowk'], data['slowd'] = ta.STOCH(data['high'], data['low'], data['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

#     data['STOCHFfastk'], data['STOCHFfastd'] = ta.STOCHF(data['high'], data['low'], data['close'], fastk_period=5, fastd_period=3, fastd_matype=0)

#     data['STOCHRSIfastk'], data['STOCHRSIfastd'] = ta.STOCHRSI(data['close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)

#     data['TRIX'] = ta.TRIX(data['close'], timeperiod=30)

#     data['ULTOSC'] = ta.ULTOSC(data['high'], data['low'], data['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)

# #### 重叠指标
#     data['BBAND_upperband'], data['BBAND_middleband'], data['BBAND_lowerband'] = ta.BBANDS(data['close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    
#     for i in range(10,201,10):
#         data['DEMA'+str(i)] = ta.DEMA(data['close'], timeperiod=30)

#         data['EMA'+str(i)] = ta.EMA(data['close'], timeperiod=30)
#         data['TEMA'+str(i)] = ta.TEMA(data['close'], timeperiod=30)

#         data['TRIMA'+str(i)] = ta.TRIMA(data['close'], timeperiod=30)

#         data['WMA'+str(i)] = ta.WMA(data['close'], timeperiod=30)


#         data['KAMA'+str(i)] = ta.KAMA(data['close'], timeperiod=30)

#         data['SMA'+str(i)] = ta.SMA(data['close'], timeperiod=30)

#         data['MA'+str(i)] = ta.MA(data['close'], timeperiod=30, matype=0)

#     for i in range(14,211, 14):         
#         data['MIDPOINT'] = ta.MIDPOINT(data['close'], timeperiod=14)

#         data['MIDPRICE'] = ta.MIDPRICE(data['high'], data['low'], timeperiod=14)

#     # data['MAMA_mama'],data['MAMA_fama'] = ta.MAMA(data['close'], fastlimit=1, slowlimit=1)

#     # data['MAVP'] = ta.MAVP(data['close'], periods, minperiod=2, maxperiod=30, matype=0)


#     data['SAR'] = ta.SAR(data['high'], data['low'], acceleration=0, maximum=0)

#     data['SAREXT'] = ta.SAREXT(data['high'], data['low'], startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)



#     data['T3'] = ta.T3(data['close'], timeperiod=5, vfactor=0)

#     data['HT_TRENDLINE'] = ta.HT_TRENDLINE(data['close'])


# #########Volume Indicator Functions
#     data['AD'] = ta.AD(data['high'], data['low'], data['close'], data['volume'])

#     data['ADSC'] = ta.ADOSC(data['high'], data['low'], data['close'], data['volume'], fastperiod=3, slowperiod=10)

#     data['OBV'] = ta.OBV(data['close'], data['volume'])

# ########Volatility Indicator Functions
#     for i in range(14,211, 14):
#         data['ATR'+str(i)] = ta.ATR(data['high'], data['low'], data['close'], timeperiod=14)

#         data['NATR'+str(i)] = ta.NATR(data['high'], data['low'], data['close'], timeperiod=14)

#     data['TRANGE'] = ta.TRANGE(data['high'], data['low'], data['close'])


# ######Cycle Indicator Functions
#     data['HT_DCPERIOD'] = ta.HT_DCPERIOD(data['close'])

#     data['HT_DCPHASE'] = ta.HT_DCPHASE(data['close'])

#     data['HT_PHASOR_inphase'], data['HT_PHASOR_quadrature'] = ta.HT_PHASOR(data['close'])

#     data['HT_SINE_sine'], data['HT_SINE_leadsine'] = ta.HT_SINE(data['close'])

#     data['HT_TRENDMODE'] = ta.HT_TRENDMODE(data['close'])
#     return data

# def data_process(target='all'):
#     data_all = pd.DataFrame()
#     data_train = pd.DataFrame()
#     data_test = pd.DataFrame()
#     row_path="/home/pc/matrad/leaf/factor/daily_data/price_data"
#     processed_path="/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/"
#     feature_path="/home/pc/matrad/leaf/factor/daily_data"
#     file_list = os.listdir(row_path)#[:50]
#     for file in tqdm(file_list):  # os.listdir('.')遍历文件夹内的每个文件名，并返回一个包含文件名的list
#         print(file)
#         data_org = pd.read_csv(row_path + "/" + file)
#         if target == 'renew':
#             data = data.iloc[-600:,:]
#         else:
#             data =data_org.iloc[-700:,:]

#         data["close"] = data["close"] * data["adjustflag"]
#         data["open"] = data["open"] * data["adjustflag"]
#         data["high"] = data["high"] * data["adjustflag"]
#         data["low"] = data["low"] * data["adjustflag"]
#         data["preclose"] = data["preclose"] * data["adjustflag"]
#         data['close_t-1'] = data["close"].shift(1)
#         data["lagRet"] = np.log(data["close"] / data["close_t-1"])
#         data.drop(columns=['close_t-1'], inplace=True)
#         data = signal_calc(data)
        
#         for i in range(10, 500, 20):
#             data["lagRet_" + str(i)] = data["lagRet"].rolling(i).sum()
#         for j in range(3, 10, 2):
#             data["ret_to_route_" + str(i)] = data["lagRet"].rolling(i).sum()
#         for k in range(10, 50, 5):
#             data["ret_to_route_" + str(k)] = data["lagRet"].rolling(k).sum()

#         for i in range(5, 30, 5):
#             data['roll_min'] = data["close"].rolling(i).min()
#             data['roll_max'] = data["close"].rolling(i).max()
#             data["rpos" + str(i)] = (data["close"] - data['roll_min'] ) / (data['roll_max'] - data['roll_min'] )
#             data.drop(columns=['roll_min','roll_max'], inplace=True)
#         for i in range(30, 120, 15):
#             data['roll_min'] = data["close"].rolling(i).min()
#             data['roll_max'] = data["close"].rolling(i).max()
#             data["rpos" + str(i)] = (data["close"] - data['roll_min'] ) / (data['roll_max'] - data['roll_min'] )
#             data.drop(columns=['roll_min','roll_max'], inplace=True)

#         # print('before', data)
#         data_obj = data.select_dtypes(exclude='number').copy()
#         data = data.select_dtypes(include='number').copy()
#         columns = np.array(data.columns)
#         columns_obj = np.array(data_obj.columns)
#         data_columns =np.concatenate([columns_obj,columns],axis = 0)
#         # print('data_columns',data_columns.shape)

#         # print('pre',data)
#         # print('data.mean(axis=1)',data.mean(axis=0))
#         data = np.divide(data.values - np.nanmean(data.values, axis=0), np.nanstd(data.values, axis=0))
#         # data = pd.DataFrame(data,columns=columns)
#         # print('after std ',data.head(5),data.shape,data_obj.head(5),data_obj.shape)
#         # # print(data.tail(5))
#         data = np.concatenate([data_obj.values,data], axis=1)
#         # print('data.shape',data.shape)
#         data = pd.DataFrame(data)
#         data.columns = data_columns
#         # print('data.shape',data.shape)
#         # print('cat', data.shape, data)
#         # data = pd.concat([data_obj,data],axis = 1)

#         # data["lable"] = np.sign(np.maximum(data["lagRet_30"].shift(-30) - np.log(1.15), 0) )
#         data["label_month_15%" ] = np.sign(np.maximum((data["lagRet"].rolling(20).sum()).shift(-20) - np.log(1.15), 0) )
#         data["label_week_7%" ] = np.sign(np.maximum((data["lagRet"].rolling(5).sum()).shift(-5) - np.log(1.07), 0) )
#         data["label_week_15%" ] = np.sign(np.maximum((data["lagRet"].rolling(5).sum()).shift(-5) - np.log(1.15), 0) )
#         data["label_month_%2" ] = np.sign(np.maximum((data["lagRet"].rolling(20).sum()).shift(-20) - np.log(1.02), 0) )
#         # print('after',data.head(10),data.tail(10),data.shape)



#         if target == 'renew':
#             data = pd.concat([data_org,data.iloc[-1,:]],axis=0)
#         data.to_csv(processed_path + "/" + file, index=False)
#         data_all=pd.concat([data,data_all],axis=0)
#         data_train= pd.concat([data.iloc[:-150,:],data_train],axis=0)
#         data_test = pd.concat([data.iloc[-150:,:],data_test],axis=0)


#     print(data_all.head(5))
#     print(data_all.tail(5))
#     data_all = data_all[(data_all.isST != 1)]  # &(dt.c1!=56)]
#     print(data[data["label_month_%2" ]==1].shape[0], data[data["label_month_%2" ]==0].shape[0])
#     data_all.drop(columns=["isST",'preclose'], inplace=True)
#     # df.dropna(axis=0, how="any", inplace=True)
    
#     # data = (data - data.mean(axis=1))/data.std(axis=1)
    

#     data_all.to_csv(feature_path + "/" + "feature_dall.csv", index=False)
#     data_train.to_csv(feature_path + "/" + "feature_dtrain.csv", index=False)
#     data_test.to_csv(feature_path + "/" + "feature_dtest.csv", index=False)



# def file_remove():
#     main_path = "/home/pc/matrad/leaf/factor/daily_data/data_processed"
#     for file in os.listdir(main_path):
#         if file[-3:] == "csv":
#             os.remove(main_path + "/" + file)


# # data_process()  # 默认处理日数据
# from multiprocess import Pool
# def mutilcore():
#     pool = Pool(9) # 制定要开启的进程数, 限定了进程上限
#     # pool.map(data_process, ('renew',))
#     pool.map(data_process, ('all',))

# if __name__ == '__main__':
#     # mutilcore()
# # data_process(target='renew')  # 默认处理日数据
#     data_process()


# file_remove()
# # if file[-2: ] == 'py':
# #     continue   #过滤掉改名的.py文件
# # name = file.replace('5day', '1day')   #去掉空格
# if file == 'rename.py' :
#   continue
# st_name = file
# for j in os.listdir(main_path+'/'+file):
#   if j in ['__pycache__','run_and_check.sh','tools.py','feature_test.txt']:
#     continue
#   if j not in [file+'.py',file+'.txt']:
#     # print(j)
#     os.chdir(main_path+'/'+file)
#     print(st_name)
#     if j[-2: ] == 'py':
#       os.rename(j, st_name +'.py')

#     if j[-3:] == 'txt':
#       os.rename(j, st_name +'.txt')
