import os
import numpy as np
import pandas as pd
import talib as ta
import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm
from multiprocessing import Pool

# This is for timing
def timer(func):
    def func_wrapper(*args,**kwargs):
        from time import time
        time_start = time()
        result = func(*args,**kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('\n{0} cost time {1} s\n'.format(func.__name__, time_spend))
        return result
    return func_wrapper



class Feature_engine():
    def __init__(self):
        self.row_path = "/home/pc/matrad/leaf/factor/daily_data/price_data"
        self.processed_path = (
            "/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/"
        )
        self.feature_path = "/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/feature_data/"
        self.label_path = "/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/label_data/"
        self.file_list = os.listdir(self.row_path)  # [:50]
        self.file_list.sort(key=lambda x: int(x[3:-4]))
        # self.file_list = self.file_list[:300]
        self.label = " "

    def price_adjust(self, data):
        data["close"] = data["close"] * data["adjustflag"]
        data["open"] = data["open"] * data["adjustflag"]
        data["high"] = data["high"] * data["adjustflag"]
        data["low"] = data["low"] * data["adjustflag"]
        data["preclose"] = data["preclose"] * data["adjustflag"].shift(-1)
        data["lagRet"] = np.log(data["close"] / data["preclose"])
        return data

    def daily_signal(self, data):
        for i in range(10, 500, 20):
            data["lagRet_" + str(i)] = data["lagRet"].rolling(i).sum()

        for i in range(5, 30, 5):
            data["roll_min"+ str(i)] = data["close"].rolling(i).min()
            data["roll_max"+ str(i)] = data["close"].rolling(i).max()
            data["rpos" + str(i)] = (data["close"] - data["roll_min"+ str(i)]) / (
                data["roll_max"+ str(i)] - data["roll_min"+ str(i)]
            )
            # data.drop(columns=["roll_min", "roll_max"], inplace=True)

        for i in range(30, 120, 15):
            data["roll_min"+ str(i)] = data["close"].rolling(i).min()
            data["roll_max"+ str(i)] = data["close"].rolling(i).max()
            data["rpos" + str(i)] = (data["close"] - data["roll_min"+ str(i)]) / (
                data["roll_max"+ str(i)] - data["roll_min"+ str(i)]
            )
            # data.drop(columns=["roll_min", "roll_max"], inplace=True)

        return data

    def momentum(self, data):
        for i in [5, 10, 20, 30, 60, 90, 120, 150]:
            ( data["bbl_upperband" + str(i)], data["bbl_middleband"+ str(i)], data["bbl_lowerband"+ str(i)], ) = ta.BBANDS(data["close"], timeperiod=i, nbdevup=2, nbdevdn=2, matype=0)
        for i in range(7, 71, 7):
            data["ADX" + str(i)] = ta.ADX( data["high"], data["low"], data["close"], timeperiod=i )

        for i in range(77, 260, 21):
            data["ADX" + str(i)] = ta.ADX( data["high"], data["low"], data["close"], timeperiod=i )

        for i in range(1, 11):
            data["APO" + str(i)] = ta.APO(
                data["close"], fastperiod=12 * i, slowperiod=26 * i, matype=0
            )
            data["aroondown" + str(i)], data["aroonup" + str(i)] = ta.AROON(
                data["high"], data["low"], timeperiod=14 * i
            )
            data["AROONOSC" + str(i)] = ta.AROONOSC(
                data["high"], data["low"], timeperiod=14 * i
            )
            data["CCI" + str(i)] = ta.CCI(
                data["high"], data["low"], data["close"], timeperiod=14 * i
            )
            data["CMO" + str(i)] = ta.CMO(data["close"], timeperiod=14 * i)
            data["DX" + str(i)] = ta.DX(
                data["high"], data["low"], data["close"], timeperiod=14 * i
            )
            data["MIF" + str(i)] = ta.MFI(
                data["high"],
                data["low"],
                data["close"],
                data["volume"],
                timeperiod=14 * i,
            )
            data["MINUS_DI" + str(i)] = ta.MINUS_DI(
                data["high"], data["low"], data["close"], timeperiod=14 * i
            )
            data["MINUS_DM" + str(i)] = ta.MINUS_DM(
                data["high"], data["low"], timeperiod=14 * i
            )
            data["PLUS_DI" + str(i)] = ta.PLUS_DI(
                data["high"], data["low"], data["close"], timeperiod=14 * i
            )
            data["PLUS_DM" + str(i)] = ta.PLUS_DM(
                data["high"], data["low"], timeperiod=14 * i
            )
            data["RSI" + str(i)] = ta.RSI(data["close"], timeperiod=14 * i)
            data["WILLR" + str(i)] = ta.WILLR(
                data["high"], data["low"], data["close"], timeperiod=14 * i
            )
            data["ADXR" + str(i)] = ta.ADXR(
                data["high"], data["low"], data["close"], timeperiod=14 * i
            )
            data["macd"+ str(i)], data["macdsignal"], data["macdhist"] = ta.MACD( data["close"], fastperiod=12*i, slowperiod=26*i, signalperiod=9*i )
            data["macd_ext"+ str(i)], data["macdsignal_ext"], data["macdhist_ext"] = ta.MACDEXT( data["close"], fastperiod=12*i, fastmatype=0, slowperiod=26*i, slowmatype=0, signalperiod=9*i, signalmatype=0, )
            data["macd_fix"+ str(i)], data["macdsignal_fix"], data["macdhist_fix"] = ta.MACDFIX( data["close"], signalperiod=9*i )
            data["PPO"+ str(i)] = ta.PPO(data["close"], fastperiod=12*i, slowperiod=26*i, matype=0)

        data["BOP"] = ta.BOP(data["open"], data["high"], data["low"], data["close"])


        for i in range(10, 201, 10):
            data["ROC" + str(i)] = ta.ROC(data["close"], timeperiod=i)

            data["ROCP" + str(i)] = ta.ROCP(data["close"], timeperiod=i)

            data["ROCR" + str(i)] = ta.ROCR(data["close"], timeperiod=i)

            data["ROCR100" + str(i)] = ta.ROCR100(data["close"], timeperiod=i)

        data["slowk"], data["slowd"] = ta.STOCH(
            data["high"],
            data["low"],
            data["close"],
            fastk_period=5,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0,
        )

        data["STOCHFfastk"], data["STOCHFfastd"] = ta.STOCHF(
            data["high"],
            data["low"],
            data["close"],
            fastk_period=5,
            fastd_period=3,
            fastd_matype=0,
        )

        data["STOCHRSIfastk"], data["STOCHRSIfastd"] = ta.STOCHRSI(
            data["close"], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0
        )

        data["TRIX"] = ta.TRIX(data["close"], timeperiod=30)

        data["ULTOSC"] = ta.ULTOSC(
            data["high"],
            data["low"],
            data["close"],
            timeperiod1=7,
            timeperiod2=14,
            timeperiod3=28,
        )

    def talib_overlap(self, data):
        (
            data["BBAND_upperband"],
            data["BBAND_middleband"],
            data["BBAND_lowerband"],
        ) = ta.BBANDS(data["close"], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)

        for i in range(10, 201, 10):
            data["DEMA" + str(i)] = ta.DEMA(data["close"], timeperiod=i)
            data["EMA" + str(i)] = ta.EMA(data["close"], timeperiod=i)
            data["TEMA" + str(i)] = ta.TEMA(data["close"], timeperiod=i)
            data["TRIMA" + str(i)] = ta.TRIMA(data["close"], timeperiod=i)
            data["WMA" + str(i)] = ta.WMA(data["close"], timeperiod=i)
            data["KAMA" + str(i)] = ta.KAMA(data["close"], timeperiod=i)
            data["SMA" + str(i)] = ta.SMA(data["close"], timeperiod=i)
            data["MA" + str(i)] = ta.MA(data["close"], timeperiod=i, matype=0)
        for i in range(14, 211, 14):
            data["MIDPOINT"+ str(i)] = ta.MIDPOINT(data["close"], timeperiod=i)
            data["MIDPRICE"+ str(i)] = ta.MIDPRICE(data["high"], data["low"], timeperiod=i)

        # data['MAMA_mama'],data['MAMA_fama'] = ta.MAMA(data['close'], fastlimit=1, slowlimit=1)
        # data['MAVP'] = ta.MAVP(data['close'], periods, minperiod=2, maxperiod=30, matype=0)
        data["SAR"] = ta.SAR(data["high"], data["low"], acceleration=0.02, maximum=0.2)
        if data["SAR"].isnull().all():
            print("data[SAR]", data["SAR"], data.shape)  # 做形状过滤
        data["SAREXT"] = ta.SAREXT(
            data["high"],
            data["low"],
            startvalue=0,
            offsetonreverse=0,
            accelerationinitlong=0.02,
            accelerationlong=0.02,
            accelerationmaxlong=0.20,
            accelerationinitshort=0.02,
            accelerationshort=0.02,
            accelerationmaxshort=0.20,
        )
        if data["SAR"].isnull().all():
            print("data[SARREXT]", data["SARREXT"], data.shape)  # 做形状过滤
        # print('data[SAR]',data['SAREXT'].values.shape)
        data["T3"] = ta.T3(data["close"], timeperiod=5, vfactor=0)
        data["HT_TRENDLINE"] = ta.HT_TRENDLINE(data["close"])
        return data

    def talib_volume(self, data):
        data["AD"] = ta.AD(data["high"], data["low"], data["close"], data["volume"])
        data["ADSC"] = ta.ADOSC(
            data["high"],
            data["low"],
            data["close"],
            data["volume"],
            fastperiod=3,
            slowperiod=10,
        )
        data["OBV"] = ta.OBV(data["close"], data["volume"])
        return data

    def talib_volatity(self, data):
        for i in range(14, 211, 14):
            data["ATR" + str(i)] = ta.ATR(
                data["high"], data["low"], data["close"], timeperiod=i
            )
            data["NATR" + str(i)] = ta.NATR(
                data["high"], data["low"], data["close"], timeperiod=i
            )

        data["TRANGE"] = ta.TRANGE(data["high"], data["low"], data["close"])
        return data

    def talib_cycle(self, data):
        data["HT_DCPERIOD"] = ta.HT_DCPERIOD(data["close"])
        data["HT_DCPHASE"] = ta.HT_DCPHASE(data["close"])
        data["HT_PHASOR_inphase"], data["HT_PHASOR_quadrature"] = ta.HT_PHASOR(
            data["close"]
        )
        data["HT_SINE_sine"], data["HT_SINE_leadsine"] = ta.HT_SINE(data["close"])
        data["HT_TRENDMODE"] = ta.HT_TRENDMODE(data["close"])
        return data

    def data_standard(self, data):
        data_obj = data.select_dtypes(exclude="number").copy()
        data = data.select_dtypes(include="number").copy()
        columns = np.array(data.columns)
        columns_obj = np.array(data_obj.columns)
        data_columns = np.concatenate([columns_obj, columns], axis=0)

        data = np.divide(
            data.values - np.nanmean(data.values, axis=0),
            np.nanstd(data.values, axis=0),
        )
        data = np.concatenate([data_obj.values, data], axis=1)
        data = pd.DataFrame(data)
        data.columns = data_columns
        return data

    def label_make(self, data):  # label制作模块分割出来，缺失值暂时不处理

        data["label_month_15%"] = np.sign(
            np.maximum((data["lagRet"].rolling(20).sum()).shift(-20) - np.log(1.15), 0)
        )
        data["label_week_7%"] = np.sign(
            np.maximum((data["lagRet"].rolling(5).sum()).shift(-5) - np.log(1.07), 0)
        )
        data["label_week_15%"] = np.sign(
            np.maximum((data["lagRet"].rolling(5).sum()).shift(-5) - np.log(1.15), 0)
        )
        data["label_month_2%"] = np.sign(
            np.maximum((data["lagRet"].rolling(20).sum()).shift(-20) - np.log(1.02), 0)
        )

        data["label_month_lagRet_reg"] = (data["lagRet"].rolling(20).sum()).shift(
            -20
        )
        data["label_week_lagRet_reg"] = (data["lagRet"].rolling(5).sum()).shift(
            -5
        )

        # print(data.tail(30))
        return data
    @timer
    def forward_feature(self, target="all"):
        data_all = pd.DataFrame()
        data_train = pd.DataFrame()
        data_test = pd.DataFrame()
        for file in tqdm(self.file_list):
            data_org = pd.read_csv(self.row_path + "/" + file, engine="python")
            if target == "renew":
                data = data.iloc[-600:, :]
            else:
                data = data_org

            data = self.price_adjust(data)
            data = self.daily_signal(data)

            for i in ["isST", "preclose", "adjustflag", "tradestatus"]:
                if i in data.columns:
                    data.drop( columns=[i], inplace=True )

            data = self.talib_overlap(data)
            data = self.talib_volatity(data)
            data = self.talib_volume(data)
            data = self.talib_cycle(data)

            # data = self.data_standard(data)


            # if target == 'renew': 需要修改
            #     data = pd.concat([data_org,data.iloc[-1,:]],axis=0)

            data.to_pickle(self.feature_path + "/feature_" + file[:-3]+'pkl')
            data_all = pd.concat([data, data_all], axis=0)
            data_train = pd.concat([data.iloc[:-150, :], data_train], axis=0)
            data_test = pd.concat([data.iloc[-150:, :], data_test], axis=0)

        # data_all.to_csv(self.processed_path + "/feature_" + "dall.csv", index=False)
        # data_train.to_csv(
        #     self.processed_path + "/feature_" + "dtrain.csv", index=False
        # )
        # data_test.to_csv(
        #     self.processed_path + "/feature_" + "dtest.csv", index=False
        # )
        data_all.to_pickle(self.processed_path + "/feature_1" + "dall.pkl")
        data_train.to_pickle( self.processed_path + "/feature_1" + "dtrain.pkl")
        data_test.to_pickle( self.processed_path + "/feature_1" + "dtest.pkl")

    @timer
    def forward_label(self, target="all"):
        data_all = pd.DataFrame()
        data_train = pd.DataFrame()
        data_test = pd.DataFrame()
        for file in tqdm(self.file_list):
            data_org = pd.read_csv(self.row_path + "/" + file)
            if target == "renew":
                data = data.iloc[-50:, :]
            else:
                data = data_org

            data = self.price_adjust(data)
            data = self.label_make(data)
            
            object_and_label = ["date", "label_month_2%", "label_month_15%", "label_week_7%", "label_week_15%", "code","label_month_lagRet_reg","label_week_lagRet_reg"]
            for i in data.columns:
                if i not in object_and_label:
                    data.drop( columns=[i], inplace=True )
  
            data.to_pickle(self.label_path + "/label_" + file[:-3]+'pkl')
            data_all = pd.concat([data, data_all], axis=0)
            data_train = pd.concat([data.iloc[:-150, :], data_train], axis=0)
            data_test = pd.concat([data.iloc[-150:, :], data_test], axis=0)

        # data_all.to_csv(self.processed_path + "/label_" + "dall.csv", index=False)
        # data_train.to_csv( self.processed_path + "/label_" + "dtrain.csv", index=False )
        # data_test.to_csv( self.processed_path + "/label_" + "dtest.csv", index=False )
        data_all.to_pickle(self.processed_path + "/label1_" + "dall.pkl")
        data_train.to_pickle( self.processed_path + "/label1_" + "dtrain.pkl")
        data_test.to_pickle( self.processed_path + "/label1_" + "dtest.pkl")

    @timer
    def feature_and_label_forward(self, target="all"):
        data_all = pd.DataFrame()
        data_train = pd.DataFrame()
        data_test = pd.DataFrame()
        for file in tqdm(self.file_list):
            data_org = pd.read_csv(self.row_path + "/" + file, engine="python")
            if target == "renew":
                data = data.iloc[-600:, :]
            else:
                data = data_org

            data = self.price_adjust(data)
            data = self.daily_signal(data)

            for i in ["isST", "preclose", "adjustflag", "tradestatus"]:
                if i in data.columns:
                    data.drop( columns=[i], inplace=True )

            data = self.talib_overlap(data)
            data = self.talib_volatity(data)
            data = self.talib_volume(data)
            data = self.talib_cycle(data)
            # data = self.data_standard(data)
            data = self.label_make(data)


            # if target == 'renew': #需要修改
            #     data = pd.concat([data_org,data.iloc[-1,:]],axis=0)

            data.to_pickle(self.feature_path + "/feature_" + file[:-3]+'pkl')
            data_all = pd.concat([data, data_all], axis=0)
            data_train = pd.concat([data.iloc[:-150, :], data_train], axis=0)
            data_test = pd.concat([data.iloc[-150:, :], data_test], axis=0)

        data_all.to_csv(self.processed_path + "/feature_label_" + "dall.csv", index=False)
        data_train.to_csv(
            self.processed_path + "/feature_label_" + "dtrain.csv", index=False
        )
        data_test.to_csv(
            self.processed_path + "/feature_label_" + "dtest.csv", index=False
        )
        data_all.to_pickle(self.processed_path + "/feature_label_" + "dall.pkl")
        data_train.to_pickle( self.processed_path + "/feature_label_" + "dtrain.pkl")
        data_test.to_pickle( self.processed_path + "/feature_label_" + "dtest.pkl")

    @timer
    def org_data_make(self):
        data_org_all = pd.DataFrame()
        for file in tqdm(self.file_list):
            data = pd.read_csv(self.row_path + "/" + file)
            data_org_all = pd.concat([data, data_org_all], axis=0)
        data_org_all.to_csv(self.processed_path + "/data_org_all.csv", index=False)

    @timer
    def data_cat(self):
        for file in tqdm(self.file_list):#使用时需修改
            labels = pd.read_csv(self.label_path + "label_" + file)
            features = pd.read_csv(self.feature_path + "feature_" + file)

    @timer
    def all_cat(self, labels=["label_month_lagRet_reg"], target="all"):
        label = pd.read_pickle(self.processed_path + "label1_" + "d" + target + ".pkl")#,nrows=50000)
        label_select = label[['date', 'code', labels]]
        feature = pd.read_pickle(self.processed_path + "feature_1" + "d" + target + ".pkl")#,nrows=50000)
        data = pd.merge(feature, label_select, on=['date','code'], how='outer')
        # data = pd.concat([feature, label[labels]],axis=1)
        return data
        # data.to_csv(self.processed_path + target + ".csv")
        #保存
    @timer
    def stocks_cat(self,root='/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/feature_data', target='feature_dall'):
        file_list = os.listdir(root)
        dataframe_all = pd.DataFrame()
        for file in tqdm(file_list):
            data = pd.read_csv(os.path.join(root, file))
            if target == 'feature_dall':
                dataframe_all = pd.concat([data,dataframe_all])
            elif target == 'feature_dtrain':
                dataframe_all = pd.concat([data[:-150, :], dataframe_all])
            else:
                dataframe_all = pd.concat([data[-150:, :], dataframe_all])
        dataframe_all.to_csv(os.path.join(self.processed_path, target + ".csv"))

    
        
            


# fm = Feature_engine()
# fm.feature_and_label_forward()
# fm.stocks_cat()
# fm.stocks_cat(feature_dtrain)
# fm.stocks_cat(feature_dtest)

# fm.stocks_cat(root='/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/label_data', target='label_dall')
# fm.stocks_cat(root='/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/label_data', target='label_dtrain')
# fm.stocks_cat(root='/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/label_data', target='label_dtest')
# fm.org_data_make()

# fm.forward_feature()
# fm.forward_label()
# data = pd.read_pickle('/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/feature_dtrain.pkl')
# data1 = pd.read_csv('/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/feature_dtrain.csv')
# print(data)

# fm.all_cat(target='all',labels=["label_month_lagRet_reg","label_week_lagRet_reg", "label_month_2%", "label_month_15%", 
#                                    "label_week_7%", "label_week_15%"])
# fm.all_cat(target='train',labels=["label_month_lagRet_reg","label_week_lagRet_reg", "label_month_2%", "label_month_15%", 
#                                    "label_week_7%", "label_week_15%"])
# fm.all_cat(target='test',labels=["label_month_lagRet_reg","label_week_lagRet_reg", "label_month_2%", "label_month_15%", 
#                                    "label_week_7%", "label_week_15%"])



# if __name__ == "__main__":
#     P = Pool(processes=9)
#     P.map(func=file_manuer.forward_feature(), iterable=("all",))
#     Print('feature catulating finish')
#     P.map(func=file_manuer.forward_label(), iterable=("all",))
#     Print('label catulating finish')
#     P.map(func=file_manuer.all_cat(taget="all"), iterable=("all",))
#     P.map(func=file_manuer.all_cat(taget="train"), iterable=("all",))
#     P.map(func=file_manuer.all_cat(taget="test"), iterable=("all",))
