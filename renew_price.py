import pandas as pd
import numpy as np
import baostock as bs
import datetime
sz50 = pd.read_csv('/home/pc/matrad/leaf/factor/daily_data/sz50_stocks.csv')
hs300 = pd.read_csv('/home/pc/matrad/leaf/factor/daily_data/hs300_stocks.csv')
zz500 = pd.read_csv('/home/pc/matrad/leaf/factor/daily_data/zz500_stocks.csv')
today = datetime.date.today()
oneday=datetime.timedelta(days=1) 
yesterday=today-oneday 


def catch_data(dataclass, 
terms='date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST,peTTM,pbMRQ,psTTM,adjustflag,pcfNcfTTM',
freq='d', data_path="/home/pc/matrad/leaf/factor/daily_data/price_data", start_dates=str(yesterday)):
  for i in dataclass['code']:
    pre_price= pd.read_csv('/home/pc/matrad/leaf/factor/daily_data/price_data/'+i+'.csv')
  #### 获取沪深A股历史K线数据 ####
    # 详细指标参数，参见“历史行情指标参数”章节
    if start_dates not in pre_price['date'].values:
        rs = bs.query_history_k_data(i, terms, start_date = start_dates , frequency = freq, adjustflag="2")
        print(i)

        #### 打印结果集 ####
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        new_price = pd.DataFrame(data_list, columns=rs.fields)
        result =pd.concat([pre_price,new_price],axis=0)

        #### 结果集输出到csv文件 ####   
        result.to_csv(data_path+'/'+i+".csv", index=False)
        # print(result)
    else:
      print(i,' is not need to renew')



lg = bs.login()
#获取月线
# catch_data(sz50, terms='date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg',freq='m',data_path= '/home/pc/matrad/leaf/factor/month_data/price_data')
# catch_data(zz500, terms='date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg',freq='m',data_path= '/home/pc/matrad/leaf/factor/month_data/price_data')
# catch_data(hs300, terms='date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg',freq='m',data_path= '/home/pc/matrad/leaf/factor/month_data/price_data')

#获取日线
catch_data(sz50)
catch_data(zz500)
catch_data(hs300)
#### 登出系统 ####
bs.logout()



