import os
import numpy as np
import pandas as pd
from multiprocess import Pool
import time


def concat_data(target="train",freq='d'):
    main_path="/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data"
    feature_path="/home/pc/matrad/leaf/factor/daily_data"
    df = pd.DataFrame()
    for file in os.listdir(main_path):
        df1 = pd.read_csv(main_path + "/" + file)
        # df1.dropna(axis=0, how="any", inplace=True)
        if target == "all":
            df = pd.concat([df, df1])
            print(file)
        elif target == 'test':
            df1 = df1.iloc[-50:, :]
            df = pd.concat([df, df1])
            print(file)
        else:
            df1 = df1.iloc[:-50, :]
            df = pd.concat([df, df1])
            print(file)
    # df.dropna(axis=0, how='any', inplace=True)

    # print(df.info(), "\ncolumns:", df.columns)
    # df.set_index(["code"], inplace=True)
    # df.drop(columns=["date"], inplace=True)
    df = df[(df.isST != 1)]  # &(dt.c1!=56)]
    df.drop(columns=["isST",'preclose'], inplace=True)
    # df.dropna(axis=0, how="any", inplace=True)

    df.to_csv(feature_path + "/" + "feature_" + freq + target + ".csv", index=False)
    return df


# 定义偏函数，并传入均值
# concat_data(target = 'train')
# concat_data(target = 'all')
# concat_data(target = 'test')
# 执行map，传入列表


from multiprocess import Pool
import threading
# def mutilcore():
#     pool = Pool(9) # 制定要开启的进程数, 限定了进程上限
#     pool.map(concat_data, ('all','d',))
#     pool.map(concat_data, ('train','d',))
#     pool.map(concat_data, ('test','d',))
def multithred():
    t1 = threading.Thread(target=concat_data, args=('all','d',))
    t1.start()
    t2 = threading.Thread(target=concat_data, args=('train','d',))
    t2.start()
    t3 = threading.Thread(target=concat_data, args=('test','d',))
    t3.start()

if __name__ == '__main__':
    start = time.time()
    # mutilcore()
    # multithred()
    concat_data(target = 'train')
    concat_data(target = 'all')
    concat_data(target = 'test')
    end =time.time()
    print('time consume',end - start)

# from pathos.multiprocessing import ProcessingPool as newpool
# from pathos import multiprocessing

# cores = multiprocessing.cpu_count()
# pool = newpool(processes=cores)


# def func(x, y):
#     return x+y

# x = [1, 3, 5]
# y = [0, 7, 2]

# print(pool.map(func, x, y))

# for yi in pool.imap(func, x, y):
#     print(yi)
#     print('woniuche')
