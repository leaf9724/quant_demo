import pandas as pd
import numpy as np
import seaborn as sns
import copy
import time
import json
from tqdm import tqdm
from scipy.stats import skew
from sklearn.model_selection import (train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV, )
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor, GradientBoostingClassifier, ExtraTreesClassifier, )
from xgboost.sklearn import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
import lightgbm
import catboost
from catboost import CatBoostRegressor, CatBoostClassifier
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys 
sys.path.append('/home/pc/matrad/leaf/factor/quant_demo')
from  feature_manuer import Feature_engine

TODAY = datetime.date.today()
oneday = datetime.timedelta(days=5)
YESTERDAY = TODAY - oneday


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

class Reg_model(Feature_engine):
    def __init__(self,label = ''):
        super().__init__()
        self.label = label
        self.data_x_columns = []
        self.object_and_label = ["date", "label_month_2%", "label_month_15%", "label_week_7%", "label_week_15%", "code","label_month_lagRet_reg","label_week_lagRet_reg"]
        self.backtest_columns = ['date', 'code', 'open', 'close', 'high', 'low', 'lgb', 'true']
        # self.model_name = model_name


    def predict_data_pare(self, today=YESTERDAY, file_path = "/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/feature_dall.csv.csv"):
        data = self.all_cat(self.label,'all')
        df_pred = data[data["date"] == str(today)].copy()
        data.dropna(axis=0, how="any", inplace=True)
        data_y = data[self.label]
        data_x = data.drop(columns=self.object_and_label).copy()
        self.data_x_columns = data_x.columns
        return data_x.values, data_y.values, df_pred

    def backtest_data_prepare(self, file_path = "/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/feature_dtrain.csv"):
        # data = self.all_cat(self.label,'train')
        data = pd.read_csv('/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/feature_label_dtrain.csv')
        data_y = data[self.label]
        for i in self.object_and_label:
            if i in data.columns:
                data = data.drop(columns=[i])
        data_x = data
        self.data_x_columns = data_x.columns
        return data_x.values, data_y.values

    def data_split(self, data_x, data_y):
        standard = StandardScaler()
        data_x = standard.fit_transform(data_x)
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=2022)
        return x_train, x_test, y_train, y_test

    def model_lgb(self, x_train, x_test, y_train, y_test):
        lgb_train = lightgbm.Dataset(x_train, label=y_train)
        lgb_eval = lightgbm.Dataset(x_test, label=y_test, reference=lgb_train)
        parameters = {
            "task": "train",
            "max_depth": 10,
            "min_child_weight": 5,
            "boosting_type": "goss",
            "num_leaves": 20,  # 叶子节点数
            "n_estimators": 500,
            "objective": "regression",
            "metric": {'l2_root','rmse'},
            "learning_rate": 0.2,
            "feature_fraction": 0.8,  # 小于 1.0, LightGBM 将会在每次迭代中随机选择部分特征.
            "bagging_fraction": 1,  # 类似于 feature_fraction, 但是它将在不进行重采样的情况下随机选择部分数据
            "bagging_freq": 5,  # bagging 的频率, 0 意味着禁用 bagging. k 意味着每 k 次迭代执行bagging
            "lambda_l1": 0.5,
            "lambda_l2": 0,
            "cat_smooth": 10,  # 用于分类特征,这可以降低噪声在分类特征中的影响, 尤其是对数据很少的类别
            "is_unbalance": True,  # 适合二分类。这里如果设置为True，评估结果降低3个点
            "verbose": 0,
            "class_weight": "balance",
            "n_jobs": 9,
        }

        evals_result = {}  # 记录训练结果所用
        gbm_model = lightgbm.train(
            parameters,
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            num_boost_round=800,  # 提升迭代的次数
            early_stopping_rounds=100,
            evals_result=evals_result,
            verbose_eval=50,
        )
        # from sklearn.externals import joblib
        # joblib.dump(model_lgb, '/home/pc/matrad/leaf/factor/quant_demo/sklearn_model/lgb_clf_model.pkl')
        # # clf = joblib.load('dota_model.pkl')

        return gbm_model, evals_result

    def model_eval_regression(self, gbm_model, evals_result, x_test, y_test):
        prediction = gbm_model.predict(x_test, num_iteration=gbm_model.best_iteration)
        try:
            mse = mean_squared_error(y_test, prediction)
            print("mean_squared_error:", mse)
        except:
            pass

        ax = lightgbm.plot_metric(evals_result, metric="rmse")  # metric的值与之前的params里面的值对应
        plt.title("metric")
        plt.savefig("/home/pc/matrad/leaf/factor/strategy/" + self.label + "lgb_metric.png")
        plt.tight_layout()
        plt.show()


        feature_names_pd = pd.DataFrame(
            {"column": self.data_x_columns, "importance": gbm_model.feature_importance(), })
        plt.figure(figsize=(10, 25))
        sns.barplot(x="importance", y="column",
                    data=feature_names_pd.sort_values(by="importance", ascending=False)[:50], )  # 按照importance的进行降排序
        plt.title("LightGBM Features")
        plt.tight_layout()
        plt.savefig("/home/pc/matrad/leaf/factor/strategy/" + self.label + "lgb_importantce.png")
        plt.show()

    def model_predict(self, gbm_model, df_pred, threshold=0.6):
        data_str = df_pred["date"]
        data_code = df_pred["code"]
        df_pred.drop(columns=self.object_and_label, inplace=True)
        standard = StandardScaler()
        X = standard.fit_transform(df_pred.values)
        y_pred = gbm_model.predict(X, num_iteration=gbm_model.best_iteration)  # lgb.predict(X)
        df_pred["date"] = data_str
        df_pred["code"] = data_code
        df_pred["lgb"] = y_pred
        df_pred = df_pred[self.backtest_columns]
        df_select = df_pred.sort_values(by="lgb",ascending=False)  # by指定按哪列排序。ascending表示是否升序
        df_select.to_csv("/home/pc/matrad/leaf/factor/strategy/" + self.label + "predite_select.csv",
                         index=False, )

    def model_backtest(self, gbm_model, threshold=0.6):
        data = self.all_cat(self.label,'test')
        data = pd.read_csv('/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/feature_label_dtest.csv')
        data = data.dropna(axis=0,subset=[self.label])
        data_str = data["date"]
        data_code = data["code"]
        y_true = data[self.label]
        for i in self.object_and_label:
            if i in data.columns:
                data.drop(columns=[i], inplace=True)
        standard = StandardScaler()
        X = standard.fit_transform(data.values)
        y_pred = gbm_model.predict(X, num_iteration=gbm_model.best_iteration)  # 修改

        # print(np.isnan(X).T.any(), np.isnan(y_prob).sum(), np.isnan(y_true).T.any())
        print("MSE:", mean_squared_error(y_true, y_pred))
        print('MAE:',mean_absolute_error(y_true, y_pred))

        data["date"] = data_str
        data["code"] = data_code
        data["lgb"] = y_pred
        data["true"] = y_true.values

        data = data[self.backtest_columns]
        data = data.sort_values(by="lgb",ascending=False)
        print(data.head(20))
        data.to_csv("/home/pc/matrad/leaf/factor/strategy/" + self.label + "_predict_to_backtest.csv",
                    index=False)

    @timer    
    def backtest_process(self):
        data_x, data_y = self.backtest_data_prepare()
        x_train, x_test, y_train, y_test = self.data_split(data_x, data_y)
        gbm_model, evals_result = self.model_lgb(x_train, x_test, y_train, y_test)
        self.model_eval_regression(gbm_model, evals_result, x_test, y_test)
        self.model_backtest(gbm_model)
    @timer
    def predict_process(self):
        data_x, data_y, df_pred = self.predict_data_pare()
        x_train, x_test, y_train, y_test = self.data_split(data_x, data_y)
        gbm_model, evals_result = self.model_lgb(x_train, x_test, y_train, y_test)
        self.model_eval_regression(gbm_model, evals_result, x_test, y_test)
        self.model_predict(gbm_model, df_pred)


@timer
def lgb_tune_param( x_train, y_train):
    lgb_params1 = {'max_depth':range(3,30,1), 'min_child_weight':range(3,20,1), }
    lgb_params2 = {'n_estimators':range(100, 5000, 100), 'learning_rate':np.linspace(0, 0.2, 20), }
    lgb_params3 = {'subsample':np.linspace(0.6, 0.9, 11), 'colsample_bytree':np.linspace(0.6, 0.8, 11) }
    lgb_params4 = { "num_leaves": range(10,30,2), "cat_smooth": range(5,20,1)}
    lgb = LGBMRegressor(learning_rate=0.02, n_estimators=100, max_depth=5, min_child_weight=1, 
                  subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1, seed=2022)

    # scoring = {'neg_mean_squared_error'}
    lgb_grid = GridSearchCV(estimator=lgb, param_grid=lgb_params1, 
                            scoring='neg_mean_absolute_error', cv=5, 
                            n_jobs=9)
    lgb_grid.fit(x_train, y_train)
    print('the best scores are:', lgb_grid.best_score_)
    print('the best params are:', lgb_grid.best_params_)
    '''
    the best scores are: -16.889389263364837
    the best params are: {'max_depth': 5, 'min_child_weight': 1}
    '''

    lgb_grid = GridSearchCV(estimator=lgb, param_grid=lgb_params2, 
                            scoring='neg_mean_absolute_error', cv=5, 
                            n_jobs=9)
    lgb_grid.fit(x_train, y_train)
    print('the best scores are:', lgb_grid.best_score_)
    print('the best params are:', lgb_grid.best_params_)
    '''
    the best scores are: -11.882901307054722
    the best params are: {'learning_rate': 0.02, 'n_estimators': 100}
    '''

    lgb_grid = GridSearchCV(estimator=lgb, param_grid=lgb_params3, 
                            scoring='neg_mean_absolute_error',  cv=5, 
                            n_jobs=9)
    lgb_grid.fit(x_train, y_train)
    print('the best scores are:', lgb_grid.best_score_)
    print('the best params are:', lgb_grid.best_params_)
    '''
    the best scores are: -16.889389263364837
    the best params are: {'colsample_bytree': 0.8, 'subsample': 0.6}
    '''

    lgb_grid = GridSearchCV(estimator=lgb, param_grid=lgb_params4, 
                            scoring='neg_mean_absolute_error',  cv=5, 
                            n_jobs=9)
    lgb_grid.fit(x_train, y_train)
    print('the best scores are:', lgb_grid.best_score_)
    print('the best params are:', lgb_grid.best_params_)

    lgb_grid = GridSearchCV(estimator=lgb, param_grid=lgb_params5, 
                            scoring='neg_mean_absolute_error',  cv=5, 
                            n_jobs=9)
    lgb_grid.fit(x_train, y_train)
    print('the best scores are:', lgb_grid.best_score_)
    print('the best params are:', lgb_grid.best_params_)


my_model = Reg_model("label_week_lagRet_reg")
# my_model.label = "label_week_lagRet_reg"
data_x, data_y = my_model.backtest_data_prepare()
x_train, x_test, y_train, y_test = my_model.data_split(data_x, data_y)
# print(x_train.shape)
my_model.backtest_process()
# my_model.predict_process()
# lgb_tune_param(x_train,y_train)

