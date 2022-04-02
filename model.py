import pandas as pd
import numpy as np
import seaborn as sns
import copy
import time
import json
from tqdm import tqdm
from scipy.stats import skew
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from xgboost.sklearn import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
import lightgbm
from catboost import CatBoostRegressor, CatBoostClassifier
import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

today = datetime.date.today()
oneday = datetime.timedelta(days=2)
yesterday = today - oneday

print(today, yesterday)


def predict(target="predict", backtest_lable="label_month_15%"):
    if target == "predict":
        file_path = "/home/pc/matrad/leaf/factor/daily_data/feature_dall.csv"
        daily_data = pd.read_csv(file_path)
        data_p = daily_data[daily_data["date"] == (yesterday)].copy()
        daily_data.dropna(axis=0, how="any", inplace=True)
        data_y = daily_data[backtest_lable]
        data_drp = daily_data.drop( columns=[ "date", "label_month_15%", "label_week_7%", "label_week_15%", "code" ] ).copy()
        # data_drp.dropna(axis=0, how="any", inplace=True)

    else:
        file_path = "/home/pc/matrad/leaf/factor/daily_data/feature_dtrain.csv"
        daily_data = pd.read_csv(file_path)
        print('daily_data',daily_data.shape)
        daily_data.dropna(axis=0, how="any", inplace=True)
        data_y = daily_data[backtest_lable]
        data_drp = daily_data.drop( columns=[ "date", "label_month_15%", "label_week_7%", "label_week_15%", "code", ] ).copy()
        # data_drp.dropna(axis=0, how="any", inplace=True)
        # data = data_drp.values

    print(data_drp.values.shape)
    print(daily_data.columns)
    data_x, data_y = data_drp.values, (data_y.values).astype(int)
    standard = StandardScaler()
    data = standard.fit_transform(data_x)
    x_train, x_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=0.3, random_state=2022
    )
    # print('y_train',y_train)

    # #调参
    # lgb_params1 = {'max_depth':range(3,19,1),
    #                'min_child_weight':range(1,15,1),
    #                }
    # lgb_params2 = {'n_estimators':range(100, 6000, 100),
    #                'learning_rate':[0.001, 0.01,0.02, 0.05, 0.08, 0.1, 0.2],
    #                }
    # lgb_params3 = {'subsample':np.linspace(0.6, 0.8, 11),
    #                'colsample_bytree':np.linspace(0.6, 0.8, 11)
    #                }
    # lgb = LGBMClassifier(learning_rate=0.02, n_estimators=100, max_depth=5, min_child_weight=1,
    #                   subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1, seed=2022)

    # lgb_grid = GridSearchCV(estimator=lgb, param_grid=lgb_params1,
    #                         scoring='roc_auc', cv=5,
    #                         n_jobs=-1)
    # lgb_grid.fit(x_train, y_train)
    # print('the best scores are:', lgb_grid.best_score_)
    # print('the best params are:', lgb_grid.best_params_)
    # '''
    # the best scores are: -16.889389263364837
    # the best params are: {'max_depth': 5, 'min_child_weight': 1}
    # '''

    # lgb_grid = GridSearchCV(estimator=lgb, param_grid=lgb_params2,
    #                         scoring='roc_auc', cv=5,
    #                         n_jobs=-1)
    # lgb_grid.fit(x_train, y_train)
    # print('the best scores are:', lgb_grid.best_score_)
    # print('the best params are:', lgb_grid.best_params_)
    # '''
    # the best scores are: -11.882901307054722
    # the best params are: {'learning_rate': 0.02, 'n_estimators': 100}
    # '''

    # lgb_grid = GridSearchCV(estimator=lgb, param_grid=lgb_params3,
    #                         scoring='roc_auc', cv=5,
    #                         n_jobs=-1)
    # lgb_grid.fit(x_train, y_train)
    # print('the best scores are:', lgb_grid.best_score_)
    # print('the best params are:', lgb_grid.best_params_)
    # '''
    # the best scores are: -16.889389263364837
    # the best params are: {'colsample_bytree': 0.8, 'subsample': 0.6}
    # '''

    # 训练
    # result=pd.DataFrame()
    # # lgb = LGBMClassifier(learning_rate=0.02, n_estimators=100, max_depth=5, min_child_weight=1,
    # #                 subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1, seed=2022, n_jobs=9)
    # lgb.fit(x_train, y_train)
    # result['lgb'] = lgb.predict(x_test)
    # result['true'] = y_test
    # accrucy = np.sum(result['lgb'].values==result['true'].values)/len(result['true'].values)
    # print('accrucy',accrucy)
    # result['accrucy']= accrucy
    # result.to_csv('/home/pc/matrad/leaf/factor/sk_model/lgb_result.csv')

    # 训练升级版
    lgb_train = lightgbm.Dataset(x_train, label=y_train)
    lgb_eval = lightgbm.Dataset(x_test, label=y_test, reference=lgb_train)
    parameters = {
        "task": "train",
        "max_depth": 15,
        "boosting_type": "gbdt",
        "num_leaves": 20,  # 叶子节点数
        "n_estimators": 50,
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.2,
        "feature_fraction": 0.7,  # 小于 1.0, LightGBM 将会在每次迭代中随机选择部分特征.
        "bagging_fraction": 1,  # 类似于 feature_fraction, 但是它将在不进行重采样的情况下随机选择部分数据
        "bagging_freq": 3,  # bagging 的频率, 0 意味着禁用 bagging. k 意味着每 k 次迭代执行bagging
        "lambda_l1": 0.5,
        "lambda_l2": 0,
        "cat_smooth": 10,  # 用于分类特征,这可以降低噪声在分类特征中的影响, 尤其是对数据很少的类别
        "is_unbalance": False,  # 适合二分类。这里如果设置为True，评估结果降低3个点
        "verbose": 0,
        "n_jobs": 9,
    }

    evals_result = {}  # 记录训练结果所用
    gbm_model = lightgbm.train(
        parameters,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        num_boost_round=50,  # 提升迭代的次数
        early_stopping_rounds=5,
        evals_result=evals_result,
        verbose_eval=10,
    )

    result = pd.DataFrame()
    prediction = gbm_model.predict(x_test, num_iteration=gbm_model.best_iteration)
    result["lgb"] = prediction
    result["true"] = y_test
    result.to_csv( "/home/pc/matrad/leaf/factor/sk_model/" + backtest_lable + "lgb_result.csv" )
    from sklearn.metrics import roc_auc_score

    roc_auc_score = roc_auc_score(y_test, prediction)
    print("roc_auc_score:", roc_auc_score)
    prediction = np.sign(np.maximum(0, prediction - 0.6))
    print("accrucy in test", (prediction == y_test).sum() / y_test.shape[0])
    print("evals_result", evals_result)
    # return gbm_model,evals_result

    # 可视化评估
    # model,evals_result = train_model(feature_train,label)
    ax = lightgbm.plot_metric(evals_result, metric="auc")  # metric的值与之前的params里面的值对应
    plt.title("metric")
    plt.savefig( "/home/pc/matrad/leaf/factor/strategy/" + backtest_lable + "lgb_metric.png" )
    plt.show()
    # print("(daily_data.columns)[:-1]", (daily_data.columns)[:-1])
    feature_names_pd = pd.DataFrame( { "column": (data_drp.columns), "importance": gbm_model.feature_importance(), } )
    plt.figure(figsize=(10, 15))
    sns.barplot( x="importance", y="column", data=feature_names_pd.sort_values(by="importance", ascending=False), )  # 按照importance的进行降排序
    plt.title("LightGBM Features")
    plt.savefig( "/home/pc/matrad/leaf/factor/strategy/" + backtest_lable + "lgb_importantce.png" )
    plt.tight_layout()

    # 预测
    if target == "predict":
        x_backtest = data_p
        print(data_p)
        print(data_p.info())
        data_str = x_backtest["date"]
        data_code = x_backtest["code"]
        x_backtest.drop( columns=[ "date", "label_month_15%", "label_week_7%", "label_week_15%", "code", ], inplace=True, )
        x_p = standard.fit_transform(x_backtest.values)
        print("x_p.shape", x_p.shape)
        y_p = gbm_model.predict(
            x_p, num_iteration=gbm_model.best_iteration
        )  # lgb.predict(x_p)
        print("y_p", y_p)
        y_p = np.sign(np.maximum(y_p - 0.6, 0))
        print("y_p", y_p)
        x_backtest["date"] = data_str
        x_backtest["code"] = data_code
        x_backtest["lgb"] = y_p
        x_select = x_backtest[x_backtest["lgb"] == 1].copy()
        # x_select = x_select[~x_select.index.duplicated(keep='first')]
        # backtest_pre = pd.read_csv('/home/pc/matrad/leaf/factor/strategy/predite.csv',index_col='code')
        # x_backtest=pd.concat([backtest_pre,x_slect],axis=0)
        x_backtest.to_csv( "/home/pc/matrad/leaf/factor/strategy/" + backtest_lable + "predite.csv", index=False, )
        x_select.to_csv( "/home/pc/matrad/leaf/factor/strategy/" + backtest_lable + "predite_select.csv", index=False, )
    else:  # back_test
        backtest_org = pd.read_csv( "/home/pc/matrad/leaf/factor/daily_data/feature_dtest.csv" )
        data_str = backtest_org["date"]
        data_code = backtest_org["code"]
        data_predy = backtest_org[backtest_lable]
        backtest_pre = backtest_org.drop( columns=[ "date", "label_month_15%", "label_week_7%", "label_week_15%", "code", ] ).copy()  # , inplace=True)
        x_p = standard.fit_transform(backtest_pre.values)
        print("x_p.shape", x_p.shape)
        y_p = gbm_model.predict( x_p, num_iteration=gbm_model.best_iteration )  # lgb.predict(x_p) #lgb.predict(x_p)
        print("y_p", y_p.shape)
        print('data_predy.values',data_predy.values.shape)
        # print("roc:", roc_auc_score(y_p, data_predy.values))
        y_p = np.sign(np.maximum(y_p - 0.6, 0))
        accrucy=np.sum((y_p == data_predy.values)) / x_p.shape[0]
        print("accrucy in backtes:"+backtest_lable, accrucy)
        backtest_pre["date"] = data_str
        backtest_pre["code"] = data_code
        backtest_pre["lgb"] = y_p
        backtest_pre["true"] = data_predy.values
        backtest_pre["accuracy"] = accrucy
        backtest_pre.to_csv( "/home/pc/matrad/leaf/factor/strategy/" + backtest_lable + "backtest_predite.csv", index=False, )
        backtest_table = backtest_pre[backtest_pre["lgb"] == 1].copy()
        backtest_table.to_csv( "/home/pc/matrad/leaf/factor/strategy/" + backtest_lable + "backtest_predite_select.csv", index=False, )

from multiprocess import Pool
import threading
def mutilcore():
    pool = Pool(9) # 制定要开启的进程数, 限定了进程上限
    # pool.map(predict, ('predict','label_week_7%',))
    # pool.map(predict, ('predict','label_week_15%',))
    # pool.map(predict, ('predict','label_month_15%',))
    # pool.map(predict, ('backtest','label_week_7%',))
    # pool.map(predict, ('backtest','label_week_15%',))
    # pool.map(predict, ('backtest','label_month_15%',))
if __name__ == '__main__':
    start = time.time()
    # mutilcore()
    # multithred()
    predict(target="backtest", backtest_lable="label_week_7%")
    predict(target="backtest", backtest_lable="label_week_15%")
    predict(target="backtest", backtest_lable="label_month_15%")
    predict("label_week_7%")
    predict("label_week_15%")
    predict("label_month_15%")
    end =time.time()
    print('time consume',end - start)


# from multiprocess import Pool
# pool = Pool(8)
# pool.map(predict)

# def lgb_importance():

#     model,evals_result = train_model(feature_train,label)

#     ax = lightgbm.plot_metric(evals_result, metric='auc') #metric的值与之前的params里面的值对应
#     plt.title('metric')
#     plt.savefig('/home/pc/matrad/leaf/factor/strategy/lgb_metric.png')
#     plt.show()

#     feature_names_pd = pd.DataFrame({'column': feature_train.columns,
#                                      'importance': model.feature_importance(),
#                                      })
#     plt.figure(figsize=(10, 15))
#     sns.barplot(x="importance", y="column", data=feature_names_pd.sort_values(by="importance", ascending=False))  #按照importance的进行降排序
#     plt.title('LightGBM Features')
#     plt.savefig('/home/pc/matrad/leaf/factor/strategy/lgb_importantce.png')
#     plt.tight_layout()


# lgb_importance()
