import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
import xgboost as xgb
    
def mainTest():
    train_set = pd.read_csv('train.csv')
    test_set = pd.read_csv('test.csv')
    
    outliers_id = np.array([1299,524,633,1325,31,463,969,411])
    outliers_id = outliers_id - 1 # id starts with 1, index starts with 0
    train_set = train_set.drop(outliers_id)
    train_set.index = range(len(train_set))
    
    data = pd.concat([train_set, test_set], ignore_index=True)
    # DROP
    data = data.drop("SalePrice", 1)
    data = data.drop("Id", 1)
    data = data.drop('MoSold', axis=1)
    data = data.drop('MiscFeature', axis=1)
    data = data.drop('MSSubClass', axis=1)
    
    data = data.fillna(0)
    
    data = data.replace({'CentralAir': {'Y': 1, 'N': 0}})
    
    Neighborhood_Good = pd.DataFrame(np.zeros((data.shape[0], 1)), columns=['Neighborhood_Good'])
    Neighborhood_Good[data.Neighborhood == 'NridgHt'] = 1
    Neighborhood_Good[data.Neighborhood == 'StoneBr'] = 1
    Neighborhood_Good[data.Neighborhood == 'NoRidge'] = 1

    data['Neighborhood_Good'] = Neighborhood_Good
    
    train_set = train_set.replace({'PavedDrive': {'Y': 1, 'P': 0, 'N': 0}})
    train_set = train_set.replace({'MSSubClass': {20: 'SubClass_20',
                                            30: 'SubClass_30',
                                            40: 'SubClass_40',
                                            45: 'SubClass_45',
                                            50: 'SubClass_50',
                                            60: 'SubClass_60',
                                            70: 'SubClass_70',
                                            75: 'SubClass_75',
                                            80: 'SubClass_80',
                                            85: 'SubClass_85',
                                            90: 'SubClass_90',
                                           120: 'SubClass_120',
                                           150: 'SubClass_150',
                                           160: 'SubClass_160',
                                           180: 'SubClass_180',
                                           190: 'SubClass_190'}})
    
    Sale_New = pd.DataFrame(np.zeros((data.shape[0], 1)), columns=['Sale_New'])
    Sale_New[data.SaleCondition == 'Partial'] = 1
    data['Sale_New'] = Sale_New
    sqrt = ['GrLivArea']
    log1p = ['LotArea', 'LotFrontage']
    sqr = ['OverallQual']
    
    data.loc[:, sqrt] = np.sqrt(data.loc[:, sqrt])
    data.loc[:, log1p] = np.log1p(data.loc[:, log1p])
    data.loc[:, sqr] = np.square(data.loc[:, sqr])
    
    X = pd.get_dummies(data, sparse=True)
    X_train = X[:train_set.shape[0]]
    X_test = X[train_set.shape[0]:]
    
    print(X_train.shape)
    print(X_test.shape)
    
    y = np.log1p(train_set.SalePrice)
    
    model_lasso = Lasso(alpha=5e-4, max_iter=100000).fit(X_train, y)
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.2,gamma=0.0,learning_rate=0.01,max_depth=4,min_child_weight=1.5,n_estimators=7200,reg_alpha=0.9,reg_lambda=0.6,subsample=0.2,seed=42,silent=1).fit(X_train, y)
    
    p = np.expm1(model_lasso.predict(X_test))
    p_xgb = np.expm1(model_xgb.predict(X_test))
    pp = [(x+y)/2 for x,y in zip(p, p_xgb)]
    
    solution = pd.DataFrame({"Id":test_set.Id, "SalePrice":pp}, columns=['Id', 'SalePrice'])
    solution.to_csv("price_prediction.csv", index=False)
    
mainTest()
