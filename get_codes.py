import baostock as bs
import pandas as pd



    #### 登陆系统 ####
lg = bs.login()

#####################
rs = bs.query_hs300_stocks()
stocks = []
while (rs.error_code == "0") & rs.next():
    # 获取一条记录，将记录合并在一起
    stocks.append(rs.get_row_data())
result = pd.DataFrame(stocks, columns=rs.fields)
# 结果集输出到csv文件
result.to_csv( "daily_data/hs300_stocks.csv", encoding="utf-8", index=False )
print("result", result)

#######################
rs = bs.query_zz500_stocks()
stocks = []
while (rs.error_code == "0") & rs.next():
    # 获取一条记录，将记录合并在一起
    stocks.append(rs.get_row_data())
result = pd.DataFrame(stocks, columns=rs.fields)
# 结果集输出到csv文件
result.to_csv( "daily_data/zz500_stocks.csv", encoding="utf-8", index=False )
print("result", result)


########################
rs = bs.query_sz50_stocks()
stocks = []
while (rs.error_code == "0") & rs.next():
    # 获取一条记录，将记录合并在一起
    stocks.append(rs.get_row_data())
result = pd.DataFrame(stocks, columns=rs.fields)
# 结果集输出到csv文件
result.to_csv( "daily_data/sz50_stocks.csv", encoding="utf-8", index=False )
print("result", result)

#### 登出系统 ####
bs.logout()
