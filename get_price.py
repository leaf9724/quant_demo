import pandas as pd
import numpy as np
import baostock as bs
import datetime
from tqdm import tqdm

class Get_price():
    def __init__(self,):
        self.code_list = []
        self.daily_price_term = 'date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST,peTTM,pbMRQ,psTTM,adjustflag,pcfNcfTTM' 
        self.daily_data_path = "/home/pc/matrad/leaf/factor/daily_data/price_data"
        self.start_dates = '2005-01-01'
        pass

    def catch_price_data(self, code_list,     freq='d'):
        for i in tqdm(code_list):
            rs = bs.query_history_k_data(i, self.daily_price_term, start_date = self.start_dates , frequency = freq, adjustflag="2")
            print(i)
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())

            result = pd.DataFrame(data_list, columns=rs.fields)
            result.to_csv(self.daily_data_path+'/'+i+".csv", index=False) 

    def catch_profit_data(self,code_list,data_path="/home/pc/matrad/leaf/factor/daily_data/profit_data"):
        for i in tqdm(code_list):
            profit_list = []
            for year_i in range(2005,2022):
                for j in range(1,5):
                    rs_profit = bs.query_profit_data(code=i, year=year_i, quarter=j)
                    while (rs_profit.error_code == '0') & rs_profit.next():
                        profit_list.append(rs_profit.get_row_data())
                    result_profit = pd.DataFrame(profit_list, columns=rs_profit.fields)
            print(result_profit)
            result_profit.to_csv(data_path+'/'+i+".csv", encoding="gbk", index=False)

    def catch_operation_data(self,code_list,data_path="/home/pc/matrad/leaf/factor/daily_data/operation_data"):
        for i in tqdm(code_list):
            operation_list = []
            for year_i in range(2005,2022):
                for j in range(1,5):
                    rs_operation = bs.query_operation_data(code=i, year=year_i, quarter=j)
                    while (rs_operation.error_code == '0') & rs_operation.next():
                        operation_list.append(rs_operation.get_row_data())
                    result_operation = pd.DataFrame(operation_list, columns=rs_operation.fields)
            print(result_operation)
            result_operation.to_csv(data_path+'/'+i+".csv", encoding="gbk", index=False)


    def catch_growth_data(self,code_list,data_path="/home/pc/matrad/leaf/factor/daily_data/growth_data"):
        for i in tqdm(code_list):
            growth_list = []
            for year_i in range(2005,2022):
                for j in range(1,5):
                    rs_growth = bs.query_growth_data(code=i, year=year_i, quarter=j)
                    while (rs_growth.error_code == '0') & rs_growth.next():
                        growth_list.append(rs_growth.get_row_data())
                    result_growth = pd.DataFrame(growth_list, columns=rs_growth.fields)
            print(result_growth)
            result_growth.to_csv(data_path+'/'+i+".csv", encoding="gbk", index=False)

    def catch_balance_data(self,code_list,data_path="/home/pc/matrad/leaf/factor/daily_data/balance_data"):
        for i in tqdm(code_list):
            balance_list = []
            for year_i in range(2005,2022):
                for j in range(1,5):
                    rs_balance = bs.query_balance_data(code=i, year=year_i, quarter=j)
                    while (rs_balance.error_code == '0') & rs_balance.next():
                        balance_list.append(rs_balance.get_row_data())
                    result_balance = pd.DataFrame(balance_list, columns=rs_balance.fields)
            print(result_balance)
            result_balance.to_csv(data_path+'/'+i+".csv", encoding="gbk", index=False)

    def catch_cash_data(self,code_list,data_path="/home/pc/matrad/leaf/factor/daily_data/cash_data"):
        for i in tqdm(code_list):
            cash_list = []
            for year_i in range(2005,2022):
                for j in range(1,5):
                    rs_cash = bs.query_cash_data(code=i, year=year_i, quarter=j)
                    while (rs_cash.error_code == '0') & rs_cash.next():
                        cash_list.append(rs_cash.get_row_data())
                    result_cash = pd.DataFrame(cash_list, columns=rs_cash.fields)
            print(result_cash)
            result_cash.to_csv(data_path+'/'+i+".csv", encoding="gbk", index=False)

    def catch_dupont_data(self,code_list,data_path="/home/pc/matrad/leaf/factor/daily_data/dupont_data"):
        for i in tqdm(code_list):
            dupont_list = []
            for year_i in range(2005,2022):
                for j in range(1,5):
                    rs_dupont = bs.query_dupont_data(code=i, year=year_i, quarter=j)
                    while (rs_dupont.error_code == '0') & rs_dupont.next():
                        dupont_list.append(rs_dupont.get_row_data())
                    result_dupont = pd.DataFrame(dupont_list, columns=rs_dupont.fields)
            print(result_dupont)
            result_dupont.to_csv(data_path+'/'+i+".csv", encoding="gbk", index=False)


    def shibor_data(self,start_dates='2005-01-01' ):
        rs = bs.query_shibor_data(start_date=start_dates ,end_date = today)
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        result.to_csv("/home/pc/matrad/leaf/factor/daily_data/shibor_data/shibor_data.csv", encoding="gbk", index=False) 

    def money_supply_data_month(self, start_dates='2005-01-01', ):
        rs = bs.query_money_supply_data_month(start_date=start_dates,end_date = today)
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        result.to_csv("/home/pc/matrad/leaf/factor/daily_data/money_supply_data_month/money_supply_data_month.csv", encoding="gbk", index=False)

sz50 = pd.read_csv('/home/pc/matrad/leaf/factor/daily_data/sz50_stocks.csv')
hs300 = pd.read_csv('/home/pc/matrad/leaf/factor/daily_data/hs300_stocks.csv')
zz500 = pd.read_csv('/home/pc/matrad/leaf/factor/daily_data/zz500_stocks.csv')

bs.login()
G = Get_price()
G.catch_price_data(sz50['code'].values)
G.catch_price_data(sz50['code'].values)
G.catch_price_data(sz50['code'].values)

# # catch_price_data(sz50, terms='date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg',freq='m',data_path= '/home/pc/matrad/leaf/factor/month_data/price_data')
# catch_price_data(zz500, terms='date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg',freq='m',data_path= '/home/pc/matrad/leaf/factor/month_data/price_data')
# catch_price_data(hs300, terms='date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg',freq='m',data_path= '/home/pc/matrad/leaf/factor/month_data/price_data')


G.catch_dupont_data(sz50)
G.catch_dupont_data(zz500)
G.catch_dupont_data(hs300)

G.catch_cash_data(sz50)
G.catch_cash_data(zz500)
G.catch_cash_data(hs300)

G.catch_balance_data(sz50)
G.catch_balance_data(zz500)
G.catch_balance_data(hs300)

G.catch_growth_data(sz50)
G.catch_growth_data(zz500)
G.catch_growth_data(hs300)

G.catch_operation_data(sz50)
G.catch_operation_data(zz500)
G.catch_operation_data(hs300)

G.catch_profit_data(sz50)
G.catch_profit_data(zz500)
G.catch_profit_data(hs300)

G.money_supply_data_month()

G.shibor_data()


# #### 登出系统 ####
bs.logout()



