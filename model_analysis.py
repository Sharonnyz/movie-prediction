import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn import linear_model
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def lr_model(X_train, y_train, X_val, y_val, X_test):
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    val_pred[val_pred < 0] = 0
    test_pred[test_pred < 0] = 0
    rmse=np.sqrt(np.mean((val_pred - y_val) ** 2))
    return {'val': val_pred, 'test': test_pred,
            'error': rmse}

def xgb_model(X_train, y_train, X_val, y_val, X_test, plot_co,verbose):
    params = {'objective': 'reg:linear',
              'eta': 0.01,
              'max_depth': 6,
              'subsample': 0.6,
              'colsample_bytree': 0.7,
              'eval_metric': 'rmse',
              'seed': random_seed,
              'silent': True,
              }

    record = dict()
    model = xgb.train(params
                      , xgb.DMatrix(X_train, y_train)
                      , 100000
                      , [(xgb.DMatrix(X_train, y_train), 'train'),
                         (xgb.DMatrix(X_val, y_val), 'valid')]
                      , verbose_eval=verbose
                      , early_stopping_rounds=500
                      , callbacks=[xgb.callback.record_evaluation(record)])

    best_idx = np.argmin(np.array(record['valid']['rmse']))
    val_pred = model.predict(xgb.DMatrix(X_val), ntree_limit=model.best_ntree_limit)
    test_pred = model.predict(xgb.DMatrix(X_test), ntree_limit=model.best_ntree_limit)
    val_pred[val_pred < 0] = 0
    test_pred[test_pred < 0] = 0
    if plot_co==1:
        xgb.plot_importance(model, max_num_features=10) # , importance_type='gain'
        plt.savefig('D:/Gra1/computing/archive/xgbfeature.png', bbox_inches='tight')
        plt.close()
    return {'val': val_pred, 'test': test_pred, 'error': record['valid']['rmse'][best_idx],
            'importance': [i for k, i in model.get_score().items()]}


def lgb_model(X_train, y_train, X_val, y_val, X_test, plot_co,verbose):
    params = {'objective': 'regression',
              'num_leaves': 30,
              'min_data_in_leaf': 20,
              'max_depth': 9,
              'learning_rate': 0.004,
              # 'min_child_samples':100,
              'feature_fraction': 0.9,
              "bagging_freq": 1,
              "bagging_fraction": 0.9,
              'lambda_l1': 0.2,
              "bagging_seed": random_seed,
              "metric": 'rmse',
              # 'subsample':.8,
              # 'colsample_bytree':.9,
              "random_state": random_seed,
              "verbosity": -1}

    record = dict()
    model = lgb.train(params
                      , lgb.Dataset(X_train, y_train)
                      , num_boost_round=100000
                      , valid_sets=[lgb.Dataset(X_val, y_val)]
                      , verbose_eval=verbose
                      , early_stopping_rounds=500
                      , callbacks=[lgb.record_evaluation(record)]
                      )
    best_idx = np.argmin(np.array(record['valid_0']['rmse']))

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    val_pred[val_pred < 0] = 0
    test_pred[test_pred < 0] = 0
    if plot_co==1:
        lgb.plot_importance(model, max_num_features=10)
        plt.savefig('D:/Gra1/computing/archive/lgbfeature.png', bbox_inches='tight')
        plt.close()
    return {'val': val_pred, 'test': test_pred, 'error': record['valid_0']['rmse'][best_idx],
            'importance': model.feature_importance('gain')}


def cat_model(X_train, y_train, X_val, y_val, X_test, verbose):
    model = CatBoostRegressor(iterations=100000,
                              learning_rate=0.004,
                              depth=5,
                              eval_metric='RMSE',
                              colsample_bylevel=0.8,
                              random_seed=random_seed,
                              bagging_temperature=0.2,
                              metric_period=None,
                              early_stopping_rounds=200)
    model.fit(X_train, y_train,
              eval_set=(X_val, y_val),
              use_best_model=True,
              verbose=False)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    val_pred[val_pred < 0] = 0
    test_pred[test_pred < 0] = 0
    return {'val': val_pred, 'test': test_pred,
            'error': model.get_best_score()['validation']['RMSE'],
            'importance': model.get_feature_importance()}


if __name__ == '__main__':
    X_train_p = pd.read_csv('D:/Gra1/computing/archive/X_train.csv')
    X_test = pd.read_csv('D:/Gra1/computing/archive/X_test.csv')
    y_train_p = pd.read_csv('D:/Gra1/computing/archive/y_train.csv')
    y_val = pd.read_csv('D:/Gra1/computing/archive/y_test.csv')
    X_train_id= pd.read_csv('D:/Gra1/computing/archive/X_train_1.csv')
    X_test_id= pd.read_csv('D:/Gra1/computing/archive/X_test_1.csv')
    random_seed = 2021
    k = 10
    # CrossValidation
    fold = list(KFold(n_splits=k, shuffle=True, random_state=random_seed).split(X_train_p))
    np.random.seed(random_seed)

    result_dict = dict()
    val_pred = np.zeros(X_train_p.shape[0])
    test_pred = np.zeros(X_test.shape[0])
    final_err = 0
    verbose = False

    df_pre = pd.DataFrame()
    df_pre['id'] = X_test_id['id']
    df_pre['revenue'] = X_test_id['revenue']


    for i, (train, val) in enumerate(fold):
        print(i + 1, "fold.    RMSE    RMLSE")

        X_train = X_train_p.loc[train, :]
        y_train = y_train_p.loc[train, :].values.ravel()
        X_val = X_train_p.loc[val, :]
        y_val = y_train_p.loc[val, :].values.ravel()


        fold_val_pred = []
        fold_test_pred = []
        fold_err = []

        plot_co=0
        if i==k-1:
            plot_co = 1
        # """ linear
        start = datetime.now()
        result = lr_model(X_train, y_train, X_val, y_val, X_test)
        if i == 0:
            df_pre['linear'] = np.expm1(result['test'])
        print("linear regression.", "{0:.5f}".format(result['error']), np.sqrt(mean_squared_log_error(result['val'], y_val)),
              '(' + str(int((datetime.now() - start).seconds)) + 's)')
        # """

        # """ xgboost
        start = datetime.now()
        result = xgb_model(X_train, y_train, X_val, y_val, X_test, plot_co,verbose)
        if i==0:
            df_pre['Xgboost'] = np.expm1(result['test'])
        fold_val_pred.append(result['val'] * 0.3)
        fold_test_pred.append(result['test'] * 0.3)
        fold_err.append(result['error'])
        print("xgb model.", "{0:.5f}".format(result['error']),np.sqrt(mean_squared_log_error(result['val'], y_val)),
              '(' + str(int((datetime.now() - start).seconds)) + 's)')
        # """

        # """ lightgbm
        start = datetime.now()
        result = lgb_model(X_train, y_train, X_val, y_val, X_test, plot_co,verbose)
        if i==0:
            df_pre['lightgbm'] = np.expm1(result['test'])
        fold_val_pred.append(result['val'] * 0.3)
        fold_test_pred.append(result['test'] * 0.3)
        fold_err.append(result['error'])
        print("lgb model.", "{0:.5f}".format(result['error']),np.sqrt(mean_squared_log_error(result['val'], y_val)),
              '(' + str(int((datetime.now() - start).seconds)) + 's)')
        # """

        # """ catboost model
        start = datetime.now()
        result = cat_model(X_train, y_train, X_val, y_val, X_test,verbose)
        if i==0:
            df_pre['catboost'] = np.expm1(result['test'])
        fold_val_pred.append(result['val'] * 0.4)
        fold_test_pred.append(result['test'] * 0.4)
        fold_err.append(result['error'])
        print("cat model.", "{0:.5f}".format(result['error']),np.sqrt(mean_squared_log_error(result['val'], y_val)),
              '(' + str(int((datetime.now() - start).seconds)) + 's)')
        # """

        # mix result of multiple models
        val_pred[val] += np.sum(np.array(fold_val_pred), axis=0)
        print(fold_test_pred)
        test_pred += np.sum(np.array(fold_test_pred), axis=0) / k
        final_err += (sum(fold_err) / len(fold_err)) / k

        print("---------------------------")
        print("avg   err.", "{0:.5f}".format(sum(fold_err) / len(fold_err)))
        print("blend err.", "{0:.5f}".format(np.sqrt(np.mean((np.sum(np.array(fold_val_pred), axis=0) - y_val) ** 2))))

        print('')

    print("total avg   err.", final_err)
    print("total blend err.", np.sqrt(np.mean((val_pred - y_train_p.values.ravel()) ** 2)))
    print(np.sqrt(mean_squared_log_error(val_pred, y_train_p.values.ravel())))

    print(np.sqrt(np.mean((test_pred - y_val) ** 2)))
    print(np.sqrt(mean_squared_log_error(test_pred, y_val)))
    df_pre['final_revenue'] = np.expm1(test_pred)
    df_pre.to_csv('D:/Gra1/computing/archive/revenue_prediction.csv', index=False)