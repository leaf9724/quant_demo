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
from sklearn.metrics import (
    mean_squared_error,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
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
import xgboost
import lightgbm
import catboost
from catboost import CatBoostRegressor, CatBoostClassifier
import datetime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.append("/home/pc/matrad/leaf/factor/quant_demo")
from feature_manuer import Feature_engine
import warnings

warnings.filterwarnings("ignore")

TODAY = datetime.date.today()
oneday = datetime.timedelta(days=2)
YESTERDAY = TODAY - oneday


def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time

        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print("\n{0} cost time {1} s\n".format(func.__name__, time_spend))
        return result

    return func_wrapper


class model(Feature_engine):
    def __init__(self, label="", model_name=""):
        super().__init__()
        self.label = label
        self.data_x_columns = []
        self.object_and_label = [ "date", "label_month_2%", "label_month_15%", "label_week_7%", "label_week_15%", "code", "label_month_lagRet_reg", "label_week_lagRet_reg" ]
        self.model_name = model_name
        self.backtest_columns = [ "date", "code", "open", "close", "high", "low",  "true", self.model_name ]
        

    def predict_data_pare(
        self,
        today=YESTERDAY,
        file_path="/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/",
    ):
        # data = pd.read_csv(file_path+'feature_dall.csv',nrows=200)
        # label_data = pd.read_csv(file_path+'label_dall.csv')
        data = self.all_cat(self.label, "all")
        df_pred = data[data["date"] == str(today)].copy()
        data.dropna(axis=0, how="any", inplace=True)
        data_y = data[self.label]
        for i in self.object_and_label:
            if i in data.columns:
                data = data.drop(columns=[i])
        data_x = data
        self.data_x_columns = data_x.columns
        return data_x.values, data_y.values.astype(int), df_pred

    def backtest_data_prepare( self, file_path="/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/", ):
        # data = pd.read_csv( "/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/feature_label_dtrain.csv" )
        data = self.all_cat(self.label,'train')
        data.dropna(axis=0, how="any", inplace=True)

        data_y = data[self.label]
        for i in self.object_and_label:
            if i in data.columns:
                data = data.drop(columns=[i])
        data_x = data
        self.data_x_columns = data_x.columns
        return data_x.values, data_y.values.astype(int)

    def data_split(self, data_x, data_y):
        standard = StandardScaler()
        data_x = standard.fit_transform(data_x)
        x_train, x_test, y_train, y_test = train_test_split(
            data_x, data_y, test_size=0.3, random_state=2022
        )
        return x_train, x_test, y_train, y_test

    def model_lgb(self, x_train, x_test, y_train, y_test):
        lgb_train = lightgbm.Dataset(x_train, label=y_train)
        lgb_eval = lightgbm.Dataset(x_test, label=y_test, reference=lgb_train)
        parameters = {
            "task": "train",
            "max_depth": 10,
            "min_child_weight": 5,
            "boosting_type": "gbdt",
            "num_leaves": 20,  # 叶子节点数
            "n_estimators": 500,
            "objective": "binary",
            "metric": {"auc", "binary_logloss", "average_precision"},
            "learning_rate": 0.2,
            "feature_fraction": 0.8,  # 小于 1.0, LightGBM 将会在每次迭代中随机选择部分特征.
            "bagging_fraction": 1,  # 类似于 feature_fraction, 但是它将在不进行重采样的情况下随机选择部分数据
            "bagging_freq": 10,  # bagging 的频率, 0 意味着禁用 bagging. k 意味着每 k 次迭代执行bagging
            "lambda_l1": 0.5,
            "lambda_l2": 0,
            "cat_smooth": 10,  # 用于分类特征,这可以降低噪声在分类特征中的影响, 尤其是对数据很少的类别
            "is_unbalance": True,  # 适合二分类。这里如果设置为True，评估结果降低3个点
            "verbose": 0,
            # "class_weight": "balance",
            "n_jobs": 9,
        }

        evals_result = {}  # 记录训练结果所用
        gbm_model = lightgbm.train(
            parameters,
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            num_boost_round=5000,  # 提升迭代的次数
            early_stopping_rounds=1000,
            evals_result=evals_result,
            verbose_eval=10,
        )

        # from sklearn.externals import joblib
        # joblib.dump(model_lgb, '/home/pc/matrad/leaf/factor/quant_demo/sklearn_model/lgb_clf_model.pkl')
        # # clf = joblib.load('dota_model.pkl')

        return gbm_model, evals_result

    def model_xgb(self, x_train, x_test, y_train, y_test):
        lgb_train = xgboost.DMatrix(x_train, label=y_train)
        lgb_eval = xgboost.DMatrix(x_test, label=y_test)
        parameters = {
                    'max_depth':15,
                      'learning_rate':0.1,
                      'n_estimators':2000,
                      'min_child_weight' :5,
                      'max_delta_step' :0,
                      'subsample' :0.8,
                      'colsample_bytree' :0.7,
                      'reg_alpha' :0,
                      'reg_lambda' :0.4,
                      'scale_pos_weight' :0.8,
                      'silent' :True,
                      'objective' :'binary:logistic',
                      'missing' :None,
                      'eval_metric' :'auc',
                      'seed' :1440,
                      'gamma' :0
        }

        evals_result = {}  # 记录训练结果所用
        gbm_model = xgboost.train(
            parameters,
            dtrain= lgb_train,
            evals=[(lgb_train, 'train'), (lgb_eval, 'valid')],
            num_boost_round=100,  # 提升迭代的次数
            early_stopping_rounds=30,
            evals_result=evals_result,
            verbose_eval=10,
        )

        # from sklearn.externals import joblib
        # joblib.dump(model_lgb, '/home/pc/matrad/leaf/factor/quant_demo/sklearn_model/lgb_clf_model.pkl')
        # # clf = joblib.load('dota_model.pkl')

        return gbm_model, evals_result

    def model_cat(self, x_train, x_test, y_train, y_test):
        model = CatBoostClassifier(
            # custom_loss=["auc"],  # default = 'logloss'
            random_seed=42,
            logging_level="Silent",
        )

        model.fit(
            x_train,
            y_train,
            # cat_features=cat_fea_idx,
            eval_set=(x_test, y_test),
            plot=True,
        )
        _ = ""
        return model, _

    def model_eval_classifier(self, gbm_model, evals_result, x_test, y_test):
        prediction = gbm_model.predict(x_test)#, num_iteration=gbm_model.best_iteration)
        try:
            roc = roc_auc_score(y_test, prediction)
            print("roc_auc_score:", roc)
        except:
            pass
        prediction = np.sign(np.maximum(0, prediction - 0.5))
        print("accuracy in test", (prediction == y_test).sum() / len(y_test))

        ax = lightgbm.plot_metric(evals_result, metric="auc")  # metric的值与之前的params里面的值对应
        plt.title("metric")
        plt.savefig("/home/pc/matrad/leaf/factor/strategy/" + self.label + "lgb_metric.png")
        plt.tight_layout()
        plt.show()

        feature_names_pd = pd.DataFrame(
            {
                "column": self.data_x_columns,
                "importance": gbm_model.feature_importance(),
            }
        )
        plt.figure(figsize=(10, 25))
        sns.barplot( x="importance", y="column", data=feature_names_pd.sort_values(by="importance", ascending=False)[:50], )  # 按照importance的进行降排序
        plt.title("LightGBM Features")
        plt.tight_layout()
        plt.savefig(
            "/home/pc/matrad/leaf/factor/strategy/" + self.label + "lgb_importantce.png"
        )
        plt.show()

    def model_predict(self, gbm_model, df_pred, threshold=0.6):
        data_str = df_pred["date"]
        data_code = df_pred["code"]
        for i in self.object_and_label:
            if i in df_pred.columns:
                df_pred = df_pred.drop(columns=[i], inplace=True)
        # data_x = data
        # df_pred.drop(columns=self.object_and_label, inplace=True)
        standard = StandardScaler()
        X = standard.fit_transform(df_pred.values)
        y_prob = gbm_model.predict( X, num_iteration=gbm_model.best_iteration )  # lgb.predict(X) y_label = np.sign(np.maximum(y_prob - threshold, 0))
        df_pred["date"] = data_str
        df_pred["code"] = data_code
        df_pred[self.model_name] = y_label
        df_pred = df_pred[self.backtest_columns]
        df_select = df_pred[df_pred[self.model_name] == 1].copy()
        df_select.to_csv( "/home/pc/matrad/leaf/factor/strategy/" + self.label + "predite_select.csv", index=False, )

    def model_backtest(self, gbm_model, threshold=0.5):
        data = self.all_cat(self.label,'test')
        # data = pd.read_csv( "/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data/feature_label_dtest.csv" )
        print('处理前预测数据量：', data.values.shape)
        data = data.dropna(axis=0, subset=[self.label])
        data_str = data["date"]
        data_code = data["code"]
        y_true = data[self.label]
        for i in self.object_and_label:
            if i in data.columns:
                data.drop(columns=[i], inplace=True)
        # print(data.shape)
        # print(data.isnull().any())
        standard = StandardScaler()
        X = standard.fit_transform(data.values)
        print("有效预测回测样本数量：", X.shape)
        if self.model_name == 'xgb':
            X = xgboost.DMatrix(X)
        y_prob = gbm_model.predict( X)#, num_iteration=gbm_model.best_iteration )  # lgb.predict(X) #lgb.predict(X)
        y_prob = y_prob.astype(np.float64)
        if self.model_name == 'cat':
            y_label = y_prob
        else:
            y_label = np.sign(np.maximum(y_prob - threshold, 0))
        print(np.isnan(X).T.any(), np.isnan(y_prob).sum(), np.isnan(y_true).T.any())
        print( "f1_score:", f1_score( y_true, y_label, labels=None, pos_label=1, average="binary", sample_weight=None, ), )
        print( "precision_score（对于预测为1的类，正确率是）:", precision_score( y_true, y_label, labels=None, pos_label=1, average="binary", ), )
        print( "recall_score(找出多少1):", recall_score( y_true, y_label, labels=None, pos_label=1, average="binary", sample_weight=None, ), )
        print( "roc_auc_score:", roc_auc_score(y_true, y_prob, average="macro", sample_weight=None), )

        accrucy = np.sum((y_label == y_true.values)) / X.shape[0]
        print("accuracy in backtest " + self.label + ": ", accrucy)

        data["date"] = data_str
        data["code"] = data_code
        data[self.model_name] = y_label
        data["true"] = y_true.values
        data["accuracy"] = accrucy
        data = data[self.backtest_columns]
        data.to_csv( "/home/pc/matrad/leaf/factor/strategy/" + self.model_name + "_" + self.label + "predict_to_backtest.csv", index=False, )
        print(data[data[self.model_name] == 1])
        print(data[data["true"] == 1])
        data_Ylabel_equal_1 = data[data[self.model_name] == 1]
        print("data_Ylabel_equal_1", data_Ylabel_equal_1.shape)
        print( 'accuracy in "lgb = 1" in backtest' + self.label + ": ", ( data_Ylabel_equal_1["true"].values == data_Ylabel_equal_1[self.model_name].values ).sum() / len(data_Ylabel_equal_1["true"].values), )

    @timer
    def backtest_process(self):
        data_x, data_y = self.backtest_data_prepare()
        x_train, x_test, y_train, y_test = self.data_split(data_x, data_y)
        print('有效训练样本量：', x_train.shape)
        if self.model_name == "lgb":
            gbm_model, evals_result = self.model_lgb(x_train, x_test, y_train, y_test)
        if self.model_name == "xgb":
            gbm_model, evals_result = self.model_xgb(x_train, x_test, y_train, y_test)
        if self.model_name == "cat":
            gbm_model, evals_result = self.model_cat(x_train, x_test, y_train, y_test)

        # self.model_eval_classifier(gbm_model, evals_result, x_test, y_test)
        self.model_backtest(gbm_model)

    @timer
    def predict_process(self):
        data_x, data_y, df_pred = self.predict_data_pare()
        x_train, x_test, y_train, y_test = self.data_split(data_x, data_y)
        gbm_model, evals_result = self.model_lgb(x_train, x_test, y_train, y_test)
        self.model_eval_classifier(gbm_model, evals_result, x_test, y_test)
        self.model_predict(gbm_model, df_pred)


@timer
def lgb_tune_param(x_train, y_train):
    lgb_params1 = {
        "max_depth": [3, 5, 6, 7, 9, 12, 15, 17, 25],
        "min_child_weight": [1, 3, 5, 7],
    }
    lgb_params2 = {
        "n_estimators": range(100, 5000, 100),
        "learning_rate": [0.01, 0.015, 0.025, 0.05, 0.1],
    }
    lgb_params3 = {
        "subsample": [0.6, 0.7, 0.8, 0.9, 1],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1],
    }
    lgb_params4 = {"num_leaves": range(10, 30, 2), "cat_smooth": range(5, 20, 1)}
    lgb = LGBMClassifier(
        learning_rate=0.02,
        n_estimators=100,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        seed=2022,
    )

    scoring = {"AUC": "roc_auc", "precision": "precision"}
    lgb_grid = GridSearchCV(
        estimator=lgb,
        param_grid=lgb_params1,
        scoring=scoring,
        refit="AUC",
        cv=5,
        n_jobs=9,
    )
    lgb_grid.fit(x_train, y_train)
    print("the best scores are:", lgb_grid.best_score_)
    print("the best params are:", lgb_grid.best_params_)
    """
    the best scores are: -16.889389263364837
    the best params are: {'max_depth': 5, 'min_child_weight': 1}
    """

    lgb_grid = GridSearchCV(
        estimator=lgb,
        param_grid=lgb_params2,
        scoring=scoring,
        refit="AUC",
        cv=5,
        n_jobs=9,
    )
    lgb_grid.fit(x_train, y_train)
    print("the best scores are:", lgb_grid.best_score_)
    print("the best params are:", lgb_grid.best_params_)
    """
    the best scores are: -11.882901307054722
    the best params are: {'learning_rate': 0.02, 'n_estimators': 100}
    """

    lgb_grid = GridSearchCV(
        estimator=lgb,
        param_grid=lgb_params3,
        scoring=scoring,
        refit="AUC",
        cv=5,
        n_jobs=9,
    )
    lgb_grid.fit(x_train, y_train)
    print("the best scores are:", lgb_grid.best_score_)
    print("the best params are:", lgb_grid.best_params_)
    """
    the best scores are: -16.889389263364837
    the best params are: {'colsample_bytree': 0.8, 'subsample': 0.6}
    """

    lgb_grid = GridSearchCV(
        estimator=lgb,
        param_grid=lgb_params4,
        scoring=scoring,
        refit="AUC",
        cv=5,
        n_jobs=9,
    )
    lgb_grid.fit(x_train, y_train)
    print("the best scores are:", lgb_grid.best_score_)
    print("the best params are:", lgb_grid.best_params_)

    lgb_grid = GridSearchCV(
        estimator=lgb,
        param_grid=lgb_params5,
        scoring=scoring,
        refit="AUC",
        cv=5,
        n_jobs=9,
    )
    lgb_grid.fit(x_train, y_train)
    print("the best scores are:", lgb_grid.best_score_)
    print("the best params are:", lgb_grid.best_params_)


fm = Feature_engine()
fm.forward_feature()
fm.forward_label()
# fm.feature_and_label_forward()

my_model = model("label_week_7%",'lgb')

# print(my_model.label)
data_x, data_y = my_model.backtest_data_prepare()
x_train, x_test, y_train, y_test = my_model.data_split(data_x, data_y)
# print(x_train.shape)
my_model.backtest_process()
# my_model.predict_process()
# lgb_tune_param(x_train,y_train)
