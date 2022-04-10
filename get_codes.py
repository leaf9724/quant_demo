import baostock as bs
import pandas as pd





#####################
def get_hs300_codes():
    rs = bs.query_hs300_stocks()
    stocks = []
    while (rs.error_code == "0") & rs.next():
        # 获取一条记录，将记录合并在一起
        stocks.append(rs.get_row_data())
    result = pd.DataFrame(stocks, columns=rs.fields)
    # 结果集输出到csv文件
    result.to_csv( "/home/pc/matrad/leaf/factor/daily_data/hs300_stocks.csv", encoding="utf-8", index=False )
    print("result", result)

#######################
def get_zz500_codes():
    rs = bs.query_zz500_stocks()
    stocks = []
    while (rs.error_code == "0") & rs.next():
        # 获取一条记录，将记录合并在一起
        stocks.append(rs.get_row_data())
    result = pd.DataFrame(stocks, columns=rs.fields)
    # 结果集输出到csv文件
    result.to_csv( "/home/pc/matrad/leaf/factor/daily_data/zz500_stocks.csv", encoding="utf-8", index=False )
    print("result", result)


########################
def get_sz50_codes():
    rs = bs.query_sz50_stocks()
    stocks = []
    while (rs.error_code == "0") & rs.next():
        # 获取一条记录，将记录合并在一起
        stocks.append(rs.get_row_data())
    result = pd.DataFrame(stocks, columns=rs.fields)
    # 结果集输出到csv文件
    result.to_csv( "/home/pc/matrad/leaf/factor/daily_data/sz50_stocks.csv", encoding="utf-8", index=False )
    print("result", result)

#### 登陆系统 ####
lg = bs.login()

get_sz50_codes()
get_hs300_codes()
get_zz500_codes()

#### 登出系统 ####
bs.logout()
