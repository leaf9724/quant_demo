import pandas as pd 
import numpy as np 
import matplotlib 
import time
import datetime
matplotlib.use('Agg')
import  matplotlib.pyplot as plt
import os
# path = r'C:\Users\Administrator\Desktop\test' 
# os.mkdir(path + './file1')
def back_test(backtest_lable='label_month_15%'):
    table_path = "/home/pc/matrad/leaf/factor/strategy/"+backtest_lable+"backtest_predite_select.csv"
    price_path = '/home/pc/matrad/leaf/factor/strategy/'+backtest_lable+'backtest_predite.csv'
    y_data = pd.read_csv(table_path)
    price_data = pd.read_csv(price_path)
    # print(y_p)
    # print(y_p[y_p['lgb']==1])
    y_data['date'] = pd.to_datetime(y_data['date'])
    y_p = y_data[y_data['open']<200]

    pnl_buy = pd.DataFrame()
    pnl_sale = pd.DataFrame()
    total_money =1000000
    money = 1000000
    profit = []
    money_list=[]
    market_value=[]
    date_list = []
    my_storyrage= pd.DataFrame(columns=y_p.columns.values)

    for name,group in y_p.groupby("date"):
        print('name',name,type(name))
        limit=(group[group['lgb']==1].shape)[0]
        # print(limit)
        # print('stroge',my_storyrage)
        # if limit<1:
        #     continue
    #sale
        print(' my_storyrage_pre', my_storyrage)
        for code  in group[group['lgb']==1]['code']:
            if  code  in my_storyrage['code'].values:
                my_storyrage[my_storyrage['code']==code]['date']= name
        print(' my_storyrage',my_storyrage)

        for code in my_storyrage['code']:
                # print('str(name)[0:10]',str(name)[0:10])
                # print(price_data[price_data['code']==code])
                # print(price_data[price_data['date']==str(name)[0:10]])
                
                price_info = price_data[price_data['date']==str(name)[0:10]].copy()
                price_info = price_info[price_info['code']==code].copy()
                # my_storyrage[my_storyrage['code']==code]['date']=price_info[price_info['code']==code]['date']
                mine = my_storyrage[my_storyrage['code']==code]
                # print('mine',mine)

                condition1 =  code not in group[group['lgb']==1]['code']
                if backtest_lable == 'label_month_15%':
                    condition2 = price_info['open'].values/mine['open'].values>=1.15
                    t1 = datetime.datetime.strptime(price_info['date'].values.item(),"%Y-%m-%d") 
                    t2 = mine['date'].values.item()
                    # print("mine['date'].values",mine['date'].values)
                    span = datetime.timedelta(days=30)
                    condition3 = (t1-t2>span)

                if backtest_lable == 'label_week_7%':
                    condition2 = price_info['open'].values/mine['open'].values>=1.07
                    t1 = datetime.datetime.strptime(price_info['date'].values.item(),"%Y-%m-%d") 
                    t2 = mine['date'].values.item()
                    # t2 = datetime.datetime.strptime(mine['date'].values.item(),"%Y-%m-%d")
                    span = datetime.timedelta(days=7)
                    # print('data print',t1,t2,t1-t2,t1-t2>span)
                    condition3 = (t1-t2>span)
                    # condition3 = (int((str(market['date']-mine['date'])[0])[0:2]))>7
                # print(condition1,condition2,condition3)
                if condition1 and (condition2 or condition3):
                    order = price_info
                    # print('sale_order',order)
                    cost =my_storyrage[my_storyrage['code']==code ]['open'].values
                    now = order['open'].values
                    money = money+order['open'].values*100
                    profit_in_sale = 100*(now-cost)
                    profit.append(profit_in_sale)
                    print('profit in sale',100*(now-cost))

                    pnl_sale = pd.concat([pnl_sale,order])
                    ind = my_storyrage[my_storyrage['code']==code].index.values #   order.index.values
                    my_storyrage = my_storyrage.drop(ind)
                    # print('money after sale',money)
    

        if limit<1:
            continue
        for code  in group[group['lgb']==1]['code']:
            #buy
            #不在持仓中
            if  code not in my_storyrage['code'].values:
                order = group[group['code']==code]
                # print('order',order)
                try:
                    # print("order['open'].values",order['open'].values)
                    price=order['open'].values[0].item()
                    
                except:
                    print('order',order)
                # print(price)
                # print(money)
            
                if money - price*100 >=0:
                    # print('buy_order',order)
                    print('buy')
                    money = money-order['open'].values*100
                    pnl_buy = pd.concat([pnl_buy,order])
                    my_storyrage = pd.concat([my_storyrage,order])
                    # print('money after buy',money)
        
        date_list.append(name)
        # print('money',money)
        money_list.append(np.array(money).item())
        market_value.append(my_storyrage['open'].sum()*100)
        # print(len(money_list),'money_list',money_list)
        # print(len(market_value),'market_value',market_value)
        # money = money.item()

    ret=pd.DataFrame()
    ret['date'] = np.array(date_list)
    ret['money']=np.array(money_list)
    ret['market_value'] = np.array(market_value)
    ret['ret']= (ret['money'] +ret['market_value'])/total_money
    plt.plot(date_list,ret['ret'].values)
    pnl_sale['profit']=profit
    plt.savefig('/home/pc/matrad/leaf/factor/strategy/'+backtest_lable+'backtest_metric.png')
    # ret['ret']= (np.array(money_list)+np.array(market_value))/total_money
    pnl_buy.to_csv('/home/pc/matrad/leaf/factor/strategy/'+backtest_lable+'pnl_buy.csv')
    pnl_sale.to_csv('/home/pc/matrad/leaf/factor/strategy/'+backtest_lable+'pnl_sale.csv')
    my_storyrage.to_csv('/home/pc/matrad/leaf/factor/strategy/'+backtest_lable+'my_storage.csv')
    ret.to_csv('/home/pc/matrad/leaf/factor/strategy/'+backtest_lable+'ret.csv')

from multiprocess import Pool
import threading
def mutilcore():
    pool = Pool(9) # 制定要开启的进程数, 限定了进程上限
    pool.map(back_test, ('label_week_7%',))
    # pool.map(back_test, ('label_week_%7',))
    # pool.map(back_test, ('label_week_%7',))

if __name__ == '__main__':
    start = time.time()
    # mutilcore()
    # multithred()
    back_test('label_week_7%')
    back_test('label_week_%15')
    back_test('label_month_15%')
    end =time.time()
    print('time consume',end - start)

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