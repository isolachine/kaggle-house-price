import matplotlib.pyplot as plt
# from sklearn import linear_model
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import Lasso
import math
from numpy import mean, size
import xgboost as xgb


# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)

def ae(actual, predicted):
    """
    Computes the absolute error.
    This function computes the absolute error between two numbers,
    or for element between a pair of lists or numpy arrays.
    Parameters
    ----------
    actual : int, float, list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double or list of doubles
            The absolute error between actual and predicted
    """
    return np.abs(np.array(actual) - np.array(predicted))

def ce(actual, predicted):
    """
    Computes the classification error.
    This function computes the classification error between two lists
    Parameters
    ----------
    actual : list
             A list of the true classes
    predicted : list
                A list of the predicted classes
    Returns
    -------
    score : double
            The classification error between actual and predicted
    """
    return (sum([1.0 for x, y in zip(actual, predicted) if x != y]) / 
            len(actual))

def mae(actual, predicted):
    """
    Computes the mean absolute error.
    This function computes the mean absolute error between two lists
    of numbers.
    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double
            The mean absolute error between actual and predicted
    """
    return np.mean(ae(actual, predicted))

def mse(actual, predicted):
    """
    Computes the mean squared error.
    This function computes the mean squared error between two lists
    of numbers.
    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double
            The mean squared error between actual and predicted
    """
    return np.mean(se(actual, predicted))

def msle(actual, predicted):
    """
    Computes the mean squared log error.
    This function computes the mean squared log error between two lists
    of numbers.
    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double
            The mean squared log error between actual and predicted
    """
    return np.mean(sle(actual, predicted))

def rmse(actual, predicted):
    """
    Computes the root mean squared error.
    This function computes the root mean squared error between two lists
    of numbers.
    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double
            The root mean squared error between actual and predicted
    """
    return np.sqrt(mse(actual, predicted))

def rmsle(actual, predicted):
    """
    Computes the root mean squared log error.
    This function computes the root mean squared log error between two lists
    of numbers.
    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double
            The root mean squared log error between actual and predicted
    """
    return np.sqrt(msle(actual, predicted))

def se(actual, predicted):
    """
    Computes the squared error.
    This function computes the squared error between two numbers,
    or for element between a pair of lists or numpy arrays.
    Parameters
    ----------
    actual : int, float, list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double or list of doubles
            The squared error between actual and predicted
    """
    return np.power(np.array(actual) - np.array(predicted), 2)

def sle(actual, predicted):
    """
    Computes the squared log error.
    This function computes the squared log error between two numbers,
    or for element between a pair of lists or numpy arrays.
    Parameters
    ----------
    actual : int, float, list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double or list of doubles
            The squared log error between actual and predicted
    """
    return (np.power(np.log(np.array(actual) + 1) - 
            np.log(np.array(predicted) + 1), 2))

def ll(actual, predicted):
    """
    Computes the log likelihood.
    This function computes the log likelihood between two numbers,
    or for element between a pair of lists or numpy arrays.
    Parameters
    ----------
    actual : int, float, list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double or list of doubles
            The log likelihood error between actual and predicted
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    err = np.seterr(all='ignore')
    score = -(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
    np.seterr(divide=err['divide'], over=err['over'],
              under=err['under'], invalid=err['invalid'])
    if type(score) == np.ndarray:
        score[np.isnan(score)] = 0
    else:
        if np.isnan(score):
            score = 0
    return score

def log_loss(actual, predicted):
    """
    Computes the log loss.
    This function computes the log loss between two lists
    of numbers.
    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double
            The log loss between actual and predicted
    """
    return np.mean(ll(actual, predicted))


 
# scorer = make_scorer(rmsle, False)
scorer = make_scorer(mean_squared_error, False)
 
def rmse_cv(model, X, y):
    return (cross_val_score(model, X, y, scoring=scorer)).mean()



def main():
#     train = pd.read_csv('train.csv')
#     test = pd.read_csv('test.csv')
    
    train_set = pd.read_csv('train.csv')
    #Save the 'Id' column
#     train_ID = train['Id']
#     test_ID = test['Id']

    #Now drop the  'Id' colum since it's unnecessary for  the prediction process.
#     train.drop("Id", axis = 1, inplace = True)
#     test.drop("Id", axis = 1, inplace = True)
    
    train_set = train_set.fillna(0)
    # OUTLIERS
    outliers_id = np.array([309,440,319,1441,1164,1187,916,5,545,810,1416,480,582,1191,1363,381,330,727,534,946,663,67,659,1023,667,1381,4,1384,561,1212,729,629,813,875,715,711,1063,917,1454,589,496,1433,411,969,463,31,1325,633,524,1299])
    outliers_id = outliers_id - 1  # id starts with 1, index starts with 0
    train_set = train_set.drop(train_set.index[outliers_id])
    train_set.index = range(len(train_set))
    
    train_set = train_set.replace({'CentralAir': {'Y': 1, 'N': 0}})
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
    
    train_set = train_set.replace({'Utilities': {'AllPub': 1, 'NoSeWa': 0, 'NoSewr': 0, 'ELO': 0},
                             'Street': {'Pave': 1, 'Grvl': 0 },
                             'FireplaceQu': {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1,
                                            'NoFireplace': 0 
                                            },
                             'Fence': {'GdPrv': 2, 
                                       'GdWo': 2, 
                                       'MnPrv': 1, 
                                       'MnWw': 1,
                                       'NoFence': 0},
                             'ExterQual': {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1
                                            },
                             'ExterCond': {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1
                                            },
                             'BsmtQual': {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1,
                                            'NoBsmt': 0},
                             'BsmtExposure': {'Gd': 3, 
                                            'Av': 2, 
                                            'Mn': 1,
                                            'No': 0,
                                            'NoBsmt': 0},
                             'BsmtCond': {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1,
                                            'NoBsmt': 0},
                             'GarageQual': {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1,
                                            'NoGarage': 0},
                             'GarageCond': {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1,
                                            'NoGarage': 0},
                             'KitchenQual': {'Ex': 5, 
                                            'Gd': 4, 
                                            'TA': 3, 
                                            'Fa': 2,
                                            'Po': 1},
                             'Functional': {'Typ': 0,
                                            'Min1': 1,
                                            'Min2': 1,
                                            'Mod': 2,
                                            'Maj1': 3,
                                            'Maj2': 4,
                                            'Sev': 5,
                                            'Sal': 6}                             
                            })
    
    train_set = train_set.drop('MiscFeature', axis=1)
    
    sqrt = ['GrLivArea']
    log1p = ['TotRmsAbvGrd', 'LotArea', 'LotFrontage']
    sqr = ['OverallQual']
    
    train_set.loc[:, sqrt] = np.sqrt(train_set.loc[:, sqrt])
    train_set.loc[:, log1p] = np.log1p(train_set.loc[:, log1p])
    train_set.loc[:, sqr] = np.square(np.square(train_set.loc[:, sqr]))




#     train_set['TotalBsmtSF'] = np.log1p(train_set['TotalBsmtSF'])
#     train_set['LotFrontage'] = np.log1p(train_set['LotFrontage'])
#     train_set['OverallCond'] = np.square(train_set['OverallCond'])
#     train_set['OverallCond'][train_set['OverallCond'] < 5] = 0
#     train_set['OverallCond'][train_set['OverallCond'] >= 5] = 1
#     print(train_set['OverallCond'].describe())
#     GoodCond = pd.DataFrame(np.zeros((train_set.shape[0], 1)), columns=['OverallCond'])
#     GoodCond[train_set['OverallCond'] < 5] = 0
#     GoodCond[train_set['OverallCond'] >= 5] = 1
# #     GoodCond[train_set['OverallCond'] == 5] = 2
# #     GoodCond[train_set['OverallCond'] == 9] = 2
#     train_set['GoodCond'] = GoodCond
#     print(train_set['OverallCond'].value_counts())
#     print(train_set['GoodCond'].value_counts())
     
    Neighborhood_Good = pd.DataFrame(np.zeros((train_set.shape[0], 1)), columns=['Neighborhood_Good'])
    Neighborhood_Good[train_set.Neighborhood == 'NridgHt'] = 1
    Neighborhood_Good[train_set.Neighborhood == 'StoneBr'] = 1
    Neighborhood_Good[train_set.Neighborhood == 'NoRidge'] = 1
    train_set['Neighborhood_Good'] = Neighborhood_Good
    
    Sale_New = pd.DataFrame(np.zeros((train_set.shape[0], 1)), columns=['Sale_New'])
    Sale_New[train_set.SaleCondition == 'Partial'] = 1
    train_set['Sale_New'] = Sale_New

    X = pd.get_dummies(train_set, sparse=True)
    X_train = X.copy()
    print(X_train.shape)
    
    y = np.log(train_set.SalePrice)
    
    X_train = X_train.drop('Id', axis=1)
    X_train = X_train.drop('SalePrice', axis=1)
    X_train = X_train.drop('MoSold', axis=1)
#     X_train = X_train.drop('MSSubClass', axis=1)
    
    kfold = KFold(2)
#     res = []
    
    yy_pred = []
    yy_pred_xgb = []
    
    for train_index, test_index in kfold.split(X_train):
        XX_train, XX_test = X_train.ix[train_index], X_train.ix[test_index]
        yy_train = y[train_index]#, yy_test = y[train_index], y[test_index]
        model_lasso = Lasso(alpha=5e-4, max_iter=1e5).fit(XX_train, yy_train)
        p_pred = model_lasso.predict(XX_test)
        model_xgb = xgb.XGBRegressor(colsample_bytree=0.2,gamma=0.0,learning_rate=0.01,max_depth=4,min_child_weight=1.5,n_estimators=7200,reg_alpha=0.9,reg_lambda=0.6,subsample=0.2,seed=42,silent=1).fit(XX_train, yy_train)
        xgb_pred = model_xgb.predict(XX_test)
        
        yy_pred.extend(p_pred)
        yy_pred_xgb.extend(xgb_pred)
#         for i in test_index:
#             yy_pred[i] = p_pred[i - test_index[0]]
#             yy_pred_xgb[i] = xgb_pred[i - test_index[0]]
            
#         print(rmsle(p_pred, np.expm1(yy_test)))
#         res.append(rmsle(p_pred, np.expm1(yy_test)))
    kk50 = [(x+y)/2 for x,y in zip(yy_pred, yy_pred_xgb)]
    kk25 = [(x/4+3*y/4) for x,y in zip(yy_pred, yy_pred_xgb)]
    kk75 = [(3*x/4+y/4) for x,y in zip(yy_pred, yy_pred_xgb)]
    print("ResLasso = ", rmsle(np.exp(yy_pred), np.exp(y)))
    print("ResXGB = ", rmsle(np.exp(yy_pred_xgb), np.exp(y)))
    print("ResMean50 = ", rmsle(np.exp(kk50), np.exp(y)))
    print("ResMean25 = ", rmsle(np.exp(kk25), np.exp(y)))
    print("ResMean75 = ", rmsle(np.exp(kk75), np.exp(y)))
#     sns.jointplot(yy_pred, np.expm1(y))
#     print("ResMean = ", np.mean(np.square(res)))
#     model_lasso = Lasso(alpha=5e-4, max_iter=50000).fit(X_train, y)
#     p_pred = np.expm1(model_lasso.predict(X_train))
#     print(rmsle(p_pred, np.expm1(y)))
    
#     y_test = label_df
#     print("XGBoost score on training set: ", rmse(y_test, y_pred))
#         
#     y_pred_xgb = regr.predict(test_df_munged)
    
    # Run prediction on training set to get a rough idea of how well it does.
    


    plt.show()
    
    
main()
