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

    def class_label_make(self, data):
        lable_data = pd.DataFrame()
        lable_data["label_month_15%" ] = np.sign(np.maximum((data["lagRet"].rolling(20).sum()).shift(-20) - np.log(1.15), 0) )
        lable_data["label_week_7%" ] = np.sign(np.maximum((data["lagRet"].rolling(5).sum()).shift(-5) - np.log(1.07), 0) )
        lable_data["label_week_15%" ] = np.sign(np.maximum((data["lagRet"].rolling(5).sum()).shift(-5) - np.log(1.15), 0) )
        lable_data["label_month_%2" ] = np.sign(np.maximum((data["lagRet"].rolling(20).sum()).shift(-20) - np.log(1.02), 0) )
        print(lable_data.head(30))
        return lable_data

    def forward(self, target='renew'):
        for file in tqdm(self.file_list):  # os.listdir('.')遍历文件夹内的每个文件名，并返回一个包含文件名的list
            # print(file)
            data_org = pd.read_csv(self.row_path + "/" + file)
            if target == 'renew':
                data = data.iloc[-600:,:]
            else:
                data =data_org
                           
            data = self.price_adjust(data)
            data = self.daily_signal(data)
            data = self.talib_overlap(data)
            data = self.talib_volatity(data)
            data = self.talib_volume(data)
            data = self.talib_cycle(data)
            class_lable_data = self.class_label_make(data)  # bug
            data = self.data_standard(data)
            data = pd.concat([data,class_lable_data], axis=1)



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
