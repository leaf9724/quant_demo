import pandas as pd 
import numpy as np 
import matplotlib 
import time
import datetime
matplotlib.use('Agg')
import  matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
# path = r'C:\Users\Administrator\Desktop\test' 
# os.mkdir(path + './file1')

class Account:

    def __init__(self,df_all, money_init, stop_loss_rate=-0.02, stop_profit_rate=0.8,
                 max_hold_period=5, target_profit = 0.15):
        self.cash = money_init  # 现金
        self.money_init = money_init
        self.my_storyrage = pd.DataFrame(columns = ['code','buy_date','buy_price','now_price','num'])
        self.pnl_buy = pd.DataFrame(columns = ['code','buy_date','buy_price','num','cost'])
        self.pnl_sale = pd.DataFrame(columns = ['code','sale_date','sale_price','buy_price','num','profit','cost'])
        self.buy_rate = 0.0003  # 买入费率
        self.buy_min = 5  # 最小买入费率
        self.sell_rate = 0.0003  # 卖出费率
        self.sell_min = 5  # 最大买入费率
        self.stamp_duty = 0.001  # 印花税
        self.max_hold_period = max_hold_period  # 最大持股周期
        self.cost = []  # 记录真实花费

        self.stop_loss_rate = stop_loss_rate  # 止损比例
        self.stop_profit_rate = stop_profit_rate  # 止盈比例

        self.victory = []  # 记录交易胜利次数


        self.cash_all = [money_init]  # 记录每天收盘后所持现金
        self.stock_value_all = [0.0]  # 记录每天收盘后所持股票的市值
        self.market_value_all = [money_init]  # 记录每天收盘后的总市值
        self.max_market_value = money_init  # 记录最大的市值情况，用来计算回撤
        self.min_after_max_makret_value = money_init  # 记录最大市值后的最小市值
        self.max_retracement = 0  # 记录最大回撤率
        self.info = []
        self.df_all = df_all

        self.result = pd.DataFrame()
        self.win_rate = 0
    # 股票买入
    def buy_stock(self, buy_date, stock_id, stock_price, buy_num):
        """
        :param buy_date: 买入日期
        :param stock_id: 买入股票的id
        :param stcok_price: 买入股票的价格
        :param buy_num: 买入股票的数量
        :return:
        """
        tmp_len = len(self.info)
        if stock_id not in self.my_storyrage['code'].values:
            # self.my_storyrage['code'].values.append(stock_id)
            # self.buy_date.append(buy_date)
            # self.stock_price.append(stock_price)
            # self.hold_day.append(1)

            # self.info.loc[tmp_len, 'code'] = stock_id
            # self.info.loc[tmp_len, 'buy_price'] = stock_price
            # self.info.loc[tmp_len, 'buy_date'] = buy_date

            # 更新市值、现金及股票价值
            tmp_money = stock_price * buy_num
            service_change = tmp_money * self.buy_rate
            if service_change < self.buy_min:
                service_change = self.buy_min
            tmp_cash = self.cash - tmp_money - service_change
            if tmp_cash > 0:
                tmp_money = stock_price * buy_num
                service_change = tmp_money * self.buy_rate
                if service_change < self.buy_min:
                    service_change = self.buy_min

                self.cash = self.cash - tmp_money - service_change
                # self.info.loc[tmp_len, 'buy_num'] = buy_num
                tmp_cost = tmp_money + service_change
                self.cost.append(tmp_cost)
                tmp_lst_pnl = pd.DataFrame({'code':stock_id,'buy_date':buy_date,'buy_price':stock_price,'num':buy_num,'cost': tmp_cost}) 
                self.pnl_buy = pd.concat([self.pnl_buy,tmp_lst_pnl])
                tmp_lst_storage = pd.DataFrame({'code':stock_id,'buy_date':buy_date,'buy_price':stock_price,'now_price':stock_price,'num':buy_num}) 
                self.my_storyrage = pd.concat([self.my_storyrage,tmp_lst_storage])
      
   



                info = str(buy_date) + '  买入 ' + stock_id + ' (' + stock_id + ') ' \
                    + str(int(buy_num)) + '股，股价：' + str(stock_price) + ',花费：' + str(round(tmp_money.item(), 2)) + ',手续费：' \
                    + str(round(np.array(service_change).item(), 2)) + '，剩余现金：' + str(round(self.cash.item(), 2))
                self.info.append(info)
                # print(info)


    def sell_stock(self, sell_date, stock_id, sell_price, sell_num):
        """
        :param sell_date: 卖出日期
        :param stock_name: 卖出股票的名字
        :param stock_id: 卖出股票的id
        :param sell_price: 卖出股票的价格
        :param sell_num: 卖出股票的数量
        :return:
        """

        if stock_id not in self.my_storyrage['code'].values:
            raise TypeError('该股票未买入')
        # idx = self.my_storyrage[self.my_storyrage['code']==stock_id].index[0]


        tmp_money = sell_num * sell_price
        service_change = tmp_money * self.sell_rate
        if service_change < self.sell_min:
            service_change = self.sell_min
        stamp_duty = self.stamp_duty * tmp_money
        self.cash = self.cash + tmp_money - service_change - stamp_duty
        service_change = stamp_duty + service_change 
        buy_price_storyrage = self.my_storyrage[self.my_storyrage['code']== stock_id]['buy_price'].values
        profit = tmp_money - service_change - buy_price_storyrage * sell_num
        tmp_lst_pnl = pd.DataFrame({'code':stock_id,'sale_date':sell_date,'sale_price':sell_price,'buy_price':buy_price_storyrage,'num':sell_num,'profit':profit,'cost':service_change}) 
        self.pnl_sale = pd.concat([self.pnl_sale,tmp_lst_pnl])

        self.my_storyrage =  self.my_storyrage[~(self.my_storyrage['code'] == stock_id)]
 

        info = str(sell_date) + '  止损卖出' + ' (' + stock_id + ') ' \
        + str(int(sell_num)) + '股, 买入价格:'+str(buy_price_storyrage) + ', 卖出股价：' + str(sell_price) +',收入：' + str(round(tmp_money.item(), 2)) + ',手续费：' \
        + str(round(service_change.item(), 2)) + '，剩余现金：' + str(round(self.cash.item(), 2)) \
        + '，最终收益：' + str(round(profit.item(), 2))

        print(info)
        self.info.append(info)
        # idx = (self.info['code'] == stock_id) & self.info['sell_date'].isna()
        # self.info.loc[idx, 'sell_date'] = sell_date
        # self.info.loc[idx, 'sell_price'] = sell_price
        # self.info.loc[idx, 'profit'] = profit

    # 更新信息
    def update(self, day):
        # 更新市值等信息
        # print('pre update ',self.my_storyrage)
        df_today = self.df_all[self.df_all['date']==day].copy()
        code_list = self.my_storyrage['code'].values
        for code in code_list:
            price_in_all = df_today[df_today['code'] == code]['close'].values.item()
            self.my_storyrage.loc[(self.my_storyrage['code'] == code), 'now_price'] = price_in_all
        # print('after update ',self.my_storyrage)

        stock_price = self.my_storyrage['now_price'].values
        stock_num = self.my_storyrage['num'].values
        self.stock_value = np.sum(stock_num * stock_price)
        self.market_value = self.cash + self.stock_value
        self.market_value_all.append(self.market_value)
        self.stock_value_all.append(self.stock_value)
        self.cash_all.append(self.cash)

        if self.max_market_value < self.market_value:
            self.max_market_value = self.market_value
            self.min_after_max_makret_value = 99999999999
        else:
            if self.min_after_max_makret_value > self.market_value:
                self.min_after_max_makret_value = self.market_value
                #  计算回撤率
                retracement = np.abs((self.max_market_value - self.min_after_max_makret_value) / self.max_market_value)
                if retracement > self.max_retracement:
                    self.max_retracement = retracement



    def result_settle(self):
        self.result['ret'] = np.array(self.market_value_all, dtype=object) / self.money_init
        self.result['cash'] = self.cash_all
        self.result['stock_value'] = self.stock_value_all
        self.result['date'] = self.df_all['date']

        profit = self.pnl_sale['profit'].values

        self.win_rate = (profit > 0).sum() / len(profit)

    def BackTest(self):
        """
        :param buy_df: 可以买入的股票，输入为DataFrame
        :param all_df: 所有股票的DataFrame
        :param index_df: 指数对应时间的df
        :return:
        """
        ###update
        for name,group in self.df_all.groupby("date"):
            #update
            print(name,'\n##################################')
            if self.my_storyrage.empty:
                pass
            else :
                self.update(name)

           #sale
            code_list = self.my_storyrage['code'].values
            
            for code  in code_list:

                condition1 =  code not in group[group['lgb']==1]['code'].values

                #condition2 & condition4
                buy_price = self.my_storyrage[self.my_storyrage['code']==code]['buy_price'].values
                now_price = self.my_storyrage[self.my_storyrage['code']==code]['now_price'].values
                condition2 =  ((now_price/buy_price)-1) > self.stop_profit_rate 
                condition4 = ((now_price/buy_price)-1) < self.stop_loss_rate

                #condition3  
                t1 = datetime.datetime.strptime(self.my_storyrage[self.my_storyrage['code']==code]['buy_date'].values.item(),"%Y-%m-%d") 
                t2 = datetime.datetime.strptime(name,"%Y-%m-%d")#时间格式问题
                span = datetime.timedelta(days=self.max_hold_period)
                condition3 = (t2-t1>span)

                if condition1 and (condition2 or condition3 or condition4):
                    self.sell_stock(sell_date = name, stock_id = code , sell_price=group[group['code']==code]['close'].values, sell_num=100)


            #buy
            for code  in group[group['lgb']==1]['code'].values:
                if  code not in self.my_storyrage['code'].values:
                    order = group[group['code']==code]
                    self.buy_stock(buy_date= name, stock_id= code, stock_price=group[group['code']==code]['close'].values, buy_num=100)

        self.result_settle()

def BackTest(# backtest_lable =  'label_month_15%',
    # STOP_PROFIT_RATE = 0.15,
    # backtest_lable =  'label_week_7%',
    # STOP_PROFIT_RATE = 0.07,
    backtest_lable =  'label_week_15%',
    STOP_PROFIT_RATE = 0.15,
    # backtest_lable =  'label_month_%2',
    # STOP_PROFIT_RATE = 0.02,
    MONEY_INIT = 100000):

# table_path = "/home/pc/matrad/leaf/factor/strategy/"+backtest_lable+"backtest_predite_select.csv"
    price_path = '/home/pc/matrad/leaf/factor/strategy/'+backtest_lable+'backtest_predite.csv'
    origin_path ='/home/pc/matrad/leaf/factor/daily_data/feature_dall_origin.csv'
    data_oringin = pd.read_csv(origin_path)
    data_all = pd.read_csv(price_path)
    data_all = data_all[['open','close','date','code','lgb','true']]
    print(data_all)
    data_test = pd.merge(data_oringin , data_all, how='right', on=('date','code'))
    data_test = data_test[['open_x','close_x','date','code','lgb','true']]
    data_test.columns = ['open','close','date','code','lgb','true']
    print(data_test)

    account = Account(df_all = data_test, money_init =MONEY_INIT, stop_profit_rate=STOP_PROFIT_RATE)
    account.BackTest()
    df_result = account.result

    df_result.to_csv('/home/pc/matrad/leaf/factor/strategy/'+backtest_lable+'backtest_metric.csv')
    plt.figure()
    print('account ifo',np.array(account.info))

    fig, ax = plt.subplots(1,1)
    ax.plot(df_result['date'].values, account.market_value_all)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=10))
    # plt.plot(df_result['date'].values, account.market_value_all)
    plt.savefig('/home/pc/matrad/leaf/factor/strategy/metric_picture/'+backtest_lable+'backtest_metric.png')
    plt.show()
    print('胜率',account.win_rate)
    print('最大回撤',account.max_retracement)

BackTest()


                    


            
        
        # for i in range(len(day_time)):
        #     day = day_time[i]
        #     tmp_idx = buy_df['day'] == day

        #     # tmp_df = buy_df.loc[tmp_idx].reset_index()
        #     tmp_df = buy_df.loc[tmp_idx].sort_values('prob', ascending=False).reset_index()

        #     # 先卖后买
        #     # ----卖股

        #     for j in range(len(self.my_storyrage['code'].values)):
        #         stock_id = self.my_storyrage['code'].values[j]
        #         # stock_name = self.stock_name[j]
        #         sell_num = self.stock_num[j]  # 假设全卖出去
        #         sell_price = self.sell_price[j]
        #         sell_kind = self.sell_kind[j]
        #         self.sell_stock(day_time[i], stock_id, sell_price, sell_num, sell_kind)

        #     # 重置
        #     self.stock_num = []
        #     self.my_storyrage['code'].values = []
        #     self.stock_name = []
        #     self.buy_date = []
        #     self.stock_price = []
        #     self.hold_day = []
        #     self.cost = []
        #     self.sell_kind = []
        #     self.sell_price = []
        #     self.buy_price = []

        #     # ----买股
        #     if len(tmp_df) != 0:
        #         for j in range(len(tmp_df)):
        #             money = self.market_value * 0.2
        #             if money > self.cash:
        #                 money = self.cash
        #             if money < 5000:  # 假设小于5000RMB，就不买股票
        #                 break

        #             buy_num = (money / tmp_df[buy_price][j]) // 100
        #             if buy_num == 0:
        #                 continue
        #             buy_num = buy_num * 100
        #             self.buy_stock(day_time[i],
        #                            tmp_df['name'][j], tmp_df[buy_price][j], buy_num)

        #             self.buy_price.append(tmp_df[buy_price][j])
        #             # 第二天卖出的价格
        #             self.sell_price.append(tmp_df['next_open'][j])
        #             if tmp_df['next_open'][j] > tmp_df['close_price'][j]:
        #                 self.sell_kind.append(1)
        #             else:
        #                 self.sell_kind.append(2)

        #     # 更新持股周期及信息
        #     self.update(day_time[i])

        # try:
        #     self.info[['buy_date', 'sell_date', 'buy_num']] = self.info[['buy_date', 'sell_date', 'buy_num']].astype(int)
        # except:
            # pass



# print(data_all.head(10))


# def back_test(backtest_lable='label_month_15%'):
#     table_path = "/home/pc/matrad/leaf/factor/strategy/"+backtest_lable+"backtest_predite_select.csv"
#     price_path = '/home/pc/matrad/leaf/factor/strategy/'+backtest_lable+'backtest_predite.csv'
#     y_data = pd.read_csv(table_path)
#     print('acu',(y_data['lgb'].values==y_data['true'].values).sum()/len(y_data['true'].values))
#     price_data = pd.read_csv(price_path)
#     # print(y_p)
#     # print(y_p[y_p['lgb']==1])
#     y_data['date'] = pd.to_datetime(y_data['date'])
#     y_p = y_data[y_data['open']<200]

#     pnl_buy = pd.DataFrame()
#     pnl_sale = pd.DataFrame()
#     total_money =100000
#     money = 100000
#     profit = []
#     money_list=[]
#     market_value=[]
#     date_list = []
#     my_storyrage= pd.DataFrame(columns=y_p.columns.values)

#     for name,group in y_p.groupby("date"):
#         print('name',name,type(name))
#         limit=(group[group['lgb']==1].shape)[0]
#         # print(limit)
#         # print('stroge',my_storyrage)
#         # if limit<1:
#         #     continue
#     #sale
#         # print(' my_storyrage_pre', my_storyrage)
#         for code  in group[group['lgb']==1]['code']:
#             if  code  in my_storyrage['code'].values:
#                 # print(code,'name',name)
#                 my_storyrage.loc[my_storyrage[(my_storyrage.code == code)].index.tolist(),'date'] = name
#                 # my_storyrage[my_storyrage['code']==code]['date']= name
#         # print(' my_storyrage',my_storyrage)

#         for code in my_storyrage['code']:
#                 # print('str(name)[0:10]',str(name)[0:10])
#                 # print(price_data[price_data['code']==code])
#                 # print(price_data[price_data['date']==str(name)[0:10]])
                
#                 price_info = price_data[price_data['date']==str(name)[0:10]].copy()
#                 price_info = price_info[price_info['code']==code].copy()
#                 # my_storyrage[my_storyrage['code']==code]['date']=price_info[price_info['code']==code]['date']
#                 mine = my_storyrage[my_storyrage['code']==code]
#                 # print('mine',mine)

#                 condition1 =  code not in group[group['lgb']==1]['code']
#                 if backtest_lable == 'label_month_15%':
#                     # print(price_info['open'].values)
#                     # print("mine['open'].values",mine['open'].values)
#                     condition2 = price_info['open'].values/mine['open'].values>=1.15
#                     t1 = datetime.datetime.strptime(price_info['date'].values.item(),"%Y-%m-%d") 
#                     t2 = mine['date'].values.item()
#                     # print("mine['date'].values",mine['date'].values)
#                     span = datetime.timedelta(days=30)
#                     condition3 = (t1-t2>span)

#                 if backtest_lable == 'label_week_7%':
#                     # print(price_info['open'].values)
#                     # print("mine['open'].values",mine['open'].values)
#                     condition2 = price_info['open'].values/mine['open'].values>=1.07
#                     t1 = datetime.datetime.strptime(price_info['date'].values.item(),"%Y-%m-%d") 
#                     t2 = mine['date'].values.item()
#                     # t2 = datetime.datetime.strptime(mine['date'].values.item(),"%Y-%m-%d")
#                     span = datetime.timedelta(days=7)
#                     # print('data print',t1,t2,t1-t2,t1-t2>span)
#                     condition3 = (t1-t2>span)

#                 if backtest_lable == 'label_week_15%':
#                     # print(price_info['open'].values)
#                     # print("mine['open'].values",mine['open'].values)
#                     condition2 = price_info['open'].values/mine['open'].values>=1.15
#                     t1 = datetime.datetime.strptime(price_info['date'].values.item(),"%Y-%m-%d") 
#                     t2 = mine['date'].values.item()
#                     # t2 = datetime.datetime.strptime(mine['date'].values.item(),"%Y-%m-%d")
#                     span = datetime.timedelta(days=30)
#                     # print('data print',t1,t2,t1-t2,t1-t2>span)
#                     condition3 = (t1-t2>span)
#                     # condition3 = (int((str(market['date']-mine['date'])[0])[0:2]))>7
#                 print(condition1 and (condition2 or condition3))
#                 if condition1 and (condition2 or condition3):
#                     order = price_info
#                     # print('sale_order',order)
#                     cost =my_storyrage[my_storyrage['code']==code ]['open'].values
#                     now = order['open'].values
#                     money = money+order['open'].values*100
#                     profit_in_sale = 100*(now-cost)
#                     profit.append(profit_in_sale)
#                     print(backtest_lable+' profit in sale',100*(now-cost), order['date'].values, order['code'].values)

#                     pnl_sale = pd.concat([pnl_sale,order])
#                     ind = my_storyrage[my_storyrage['code']==code].index.values #   order.index.values
#                     my_storyrage = my_storyrage.drop(ind)
#                     print(my_storyrage)
    

#         if limit<1:
#             continue
#         for code  in group[group['lgb']==1]['code']:
#             #buy
#             #不在持仓中
#             if  code not in my_storyrage['code'].values:
#                 order = group[group['code']==code]
#                 # print('order',order)
#                 try:
#                     # print("order['open'].values",order['open'].values)
#                     price=order['open'].values[0].item()
                    
#                 except:
#                     print('order',order)
#                 # print(price)
#                 # print(money)
            
#                 if money - price*100 >=0:
#                     # print('buy_order',order)
#                     # print('buy')
#                     money = money-order['open'].values*100
#                     pnl_buy = pd.concat([pnl_buy,order])
#                     my_storyrage = pd.concat([my_storyrage,order])
#                     # print('money after buy',money)
        
#         date_list.append(name)
#         # print('money',money)
#         money_list.append(np.array(money).item())
#         market_value.append(my_storyrage['open'].sum()*100)
#         # print(len(money_list),'money_list',money_list)
#         # print(len(market_value),'market_value',market_value)
#         # money = money.item()

#     ret=pd.DataFrame()
#     ret['date'] = np.array(date_list)
#     ret['money']=np.array(money_list)
#     ret['market_value'] = np.array(market_value)
#     ret['ret']= (ret['money'] +ret['market_value'])/total_money
#     plt.plot(date_list,ret['ret'].values)
#     pnl_sale['profit']=profit
#     plt.savefig('/home/pc/matrad/leaf/factor/strategy/'+backtest_lable+'backtest_metric.png')
#     # ret['ret']= (np.array(money_list)+np.array(market_value))/total_money
#     pnl_buy.to_csv('/home/pc/matrad/leaf/factor/strategy/'+backtest_lable+'pnl_buy.csv')
#     pnl_sale.to_csv('/home/pc/matrad/leaf/factor/strategy/'+backtest_lable+'pnl_sale.csv')
#     my_storyrage.to_csv('/home/pc/matrad/leaf/factor/strategy/'+backtest_lable+'my_storage.csv')
#     ret.to_csv('/home/pc/matrad/leaf/factor/strategy/'+backtest_lable+'ret.csv')

# from multiprocess import Pool
# import threading
# def mutilcore():
#     pool = Pool(9) # 制定要开启的进程数, 限定了进程上限
#     pool.map(back_test, ('label_week_7%',))
#     # pool.map(back_test, ('label_week_%7',))
#     # pool.map(back_test, ('label_week_%7',))

# if __name__ == '__main__':
#     start = time.time()
#     # mutilcore()
#     # multithred()
    
#     back_test('label_week_15%')

#     back_test('label_month_15%')
#     back_test('label_month_2%')
#     end =time.time()
#     print('time consume',end - start)

            # print(pnl_buy)
    #     else:
    #         close_now = group[group['code']==code]['close'].values
    #         # print('close_now',close_now)
    #         # print((pnl_buy.loc[[code]].iloc[-1,:])['close'])
    #         close_pre = ((my_storyrage[my_storyrage['code']==code]).iloc[-1,:])['close']
    #         # print('close_pre',close_pre)
    #         date_now = group[group['code']==code]['date']
    #         # print(date_now)
    #         date_pre = my_storyrage[my_storyrage['code']==code]['date']
    #         # print(date_pre)
    #         date_diff = int((str(date_now-date_pre)[0])[0:2])
    #         # print(date_diff)

    #         #持仓收益已到或时间已到
    #         if (close_now/close_pre)>1.15 or (date_now-date_pre)>30:
    #             order = group[group['code']==code]
    #             pnl_buy = pd.concat([pnl_buy,order])
    #             money = money-order['open']*100
    #             my_storyrage = pd.concat([my_storyrage,order])

    #      #sale 
    # print(pnl_buy) 



# print(pnl_buy)    
        # elif 





    # group.to_csv('/home/pc/matrad/leaf/factor/strategy/lgb_backtest_data/'+name+'.csv',index =False)
# print(y_p.groupby('code').sum())
# print(y_p)
# print(y_p[y_p['lgb']==1])