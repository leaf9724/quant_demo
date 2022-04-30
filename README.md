## get_codes
获取沪深300，中证500，上证50股票代码，储存进股票代码池。

## get_price
获取股票代码池里股票的价格信息，财务信息等。
get_price_data中获取的字段如下：'date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST,peTTM,pbMRQ,psTTM,adjustflag,pcfNcfTTM' 

## feature_manuer
特征工程，构造各类因子。
目前构造了talib库中所有的指标信息，以及部分自研因子信息。
Feature_engine中label_make函数可以依据需要的目标动态重写。

## class_model
建模预测
框架下可以套用各类sklern库中的机器学习模型。
可选择参数进行明日预测或生成待回测文件

## back_test
回测文件
可对class_model生成的待回测文件进行回测。依据不同的回测任务与下单策略，需要进行调整。



