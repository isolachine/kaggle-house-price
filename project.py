import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import Lasso

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
    train_set = pd.read_csv('train.csv')
#     test_set = pd.read_csv('test.csv')
    train_set = train_set.fillna(0)
    train_set = train_set.replace({'CentralAir': {'Y': 1, 'N': 0}})
    
    Neighborhood_Good = pd.DataFrame(np.zeros((train_set.shape[0], 1)), columns=['Neighborhood_Good'])
    Neighborhood_Good[train_set.Neighborhood == 'NridgHt'] = 1
    Neighborhood_Good[train_set.Neighborhood == 'StoneBr'] = 1
    Neighborhood_Good[train_set.Neighborhood == 'NoRidge'] = 1
#     Neighborhood_Good[train_set.Neighborhood=='BrDale'] = 0
#     Neighborhood_Good[train_set.Neighborhood=='BrkSide'] = 0
#     Neighborhood_Good[train_set.Neighborhood=='IDOTRR'] = 0
#     Neighborhood_Good[train_set.Neighborhood=='MeadowV'] = 0
#     train_set = train_set.drop('Neighborhood', axis=1)
    train_set['Neighborhood_Good'] = Neighborhood_Good
#     Zoning_Bad = pd.DataFrame(np.ones((train_set.shape[0], 1)), columns=['Zoning_Bad'])
#     Zoning_Bad[train_set.MSZoning == 'C (all)'] = 0
#     Zoning_Bad[train_set.MSZoning == 'FV'] = 2
#     train_set['Zoning_Bad'] = Zoning_Bad

    Sale_New = pd.DataFrame(np.zeros((train_set.shape[0], 1)), columns=['Sale_New'])
    Sale_New[train_set.SaleCondition == 'Partial'] = 1
    train_set['Sale_New'] = Sale_New

    X = pd.get_dummies(train_set, sparse=True)
    X_train = X[:train_set.shape[0]]
    print(list(X_train))
    
    y = np.log1p(train_set.SalePrice)
    
    X_train = X_train.drop('Id', axis=1)
    X_train = X_train.drop('SalePrice', axis=1)
    
    X_train = X_train.drop('MSSubClass', axis=1)
    
#     print(list(X_train))
#     print(X_train.shape)
    
#     alphas = [1e-4, 5e-4, 1e-3, 5e-3]
#     cv_lasso = [rmse_cv(Lasso(alpha=alpha, max_iter=50000), X_train, y) for alpha in alphas]
#     plt.figure(1)
#     pd.Series(cv_lasso, index = alphas).plot()
    kfold = KFold(100)
    res = []
    for train_index, test_index in kfold.split(X_train):
        XX_train, XX_test = X_train.ix[train_index], X_train.ix[test_index]
        yy_train, yy_test = y[train_index], y[test_index]
        model_lasso = Lasso(alpha=5e-4, max_iter=50000).fit(XX_train, yy_train)
        p_pred = np.expm1(model_lasso.predict(XX_test))
        print(rmsle(p_pred, np.expm1(yy_test)))
        res.append(rmsle(p_pred, np.expm1(yy_test)))
    
    print("Res = ", np.mean(res))
#     model_lasso = Lasso(alpha=5e-4, max_iter=50000).fit(X_train, y)
# #     predictions = cross_val_score(model_lasso, X_train, y, cv=10)
# #     print(predictions)
# #     plt.figure(2)
# #     coef = pd.Series(model_lasso.coef_, index=X_train.columns).sort_values()
# #     imp_coef = pd.concat([coef.head(10), coef.tail(10)])
# #     imp_coef.plot(kind="barh")
# #     plt.title("Coefficients in the Model")
#     # This is a good way to see how model predict data
#     plt.figure(3)
#     p_pred = np.expm1(model_lasso.predict(X_train))
#     plt.scatter(p_pred, np.expm1(y))
#     plt.plot([min(p_pred), max(p_pred)], [min(p_pred), max(p_pred)], c="red")
#     
#     print(rmsle(p_pred, np.expm1(y)))
#     plt.show()
    
def mainTest():
    train_set = pd.read_csv('train.csv')
    test_set = pd.read_csv('test.csv')
    
    outliers_id = np.array([523, 1298])
    outliers_id = outliers_id - 1 # id starts with 1, index starts with 0
    train_set = train_set.drop(train_set.index[[523,1298]])
    train_set.index = range(len(train_set))
    
    data = pd.concat([train_set, test_set], ignore_index=True)
    # DROP
    data = data.drop("SalePrice", 1)
    data = data.drop("Id", 1)
    data = data.drop('MoSold', axis=1)
    data = data.drop('MiscFeature', axis=1)
#     data = data.drop('MSSubClass', axis=1)

    data = data.replace({'CentralAir': {'Y': 1, 'N': 0}})
    data = data.replace({'PavedDrive': {'Y': 1, 'P': 0, 'N': 0}})
    data = data.replace({'MSSubClass': {20: 'SubClass_20',
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
    
    data = data.replace({'Utilities': {'AllPub': 1, 'NoSeWa': 0, 'NoSewr': 0, 'ELO': 0},
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
    data = data.fillna(0)
    
#         
#     numeric_feats = data.dtypes[data.dtypes != "object"].index
# 
#     t = data[numeric_feats].quantile(.95)
#     use_max_scater = t[t == 0].index
#     use_95_scater = t[t != 0].index
#     data[use_max_scater] = data[use_max_scater]/data[use_max_scater].max()
#     data[use_95_scater] = data[use_95_scater]/data[use_95_scater].quantile(.95)
#     

    
    sqrt = ['GrLivArea']
    log1p = ['TotRmsAbvGrd', 'LotArea']
    sqr = ['OverallQual']
    data.loc[:, sqrt] = np.sqrt(data.loc[:, sqrt])
    data.loc[:, log1p] = np.log1p(data.loc[:, log1p])
    data.loc[:, sqr] = np.square(data.loc[:, sqr])
    
    Neighborhood_Good = pd.DataFrame(np.zeros((data.shape[0], 1)), columns=['Neighborhood_Good'])
    Neighborhood_Good[data.Neighborhood == 'NridgHt'] = 1
    Neighborhood_Good[data.Neighborhood == 'StoneBr'] = 1
    Neighborhood_Good[data.Neighborhood == 'NoRidge'] = 1
    data['Neighborhood_Good'] = Neighborhood_Good
    Sale_New = pd.DataFrame(np.zeros((data.shape[0], 1)), columns=['Sale_New'])
    Sale_New[data.SaleCondition == 'Partial'] = 1
    data['Sale_New'] = Sale_New
    
    X = pd.get_dummies(data, sparse=True)
    X_train = X[:train_set.shape[0]]
    X_test = X[train_set.shape[0]:]
    
    print(X_train.shape)
    print(X_test.shape)
    
    y = np.log1p(train_set.SalePrice)
    
#     alphas = [1e-4, 5e-4, 1e-3, 5e-3]
#     cv_lasso = [rmse_cv(Lasso(alpha=alpha, max_iter=50000), X_train, y) for alpha in alphas]
#     plt.figure(1)
#     pd.Series(cv_lasso, index = alphas).plot()
#     plt.show()
     
    model_lasso = Lasso(alpha=5e-4, max_iter=1e5).fit(X_train, y)
    p = np.expm1(model_lasso.predict(X_test))
    solution = pd.DataFrame({"Id":test_set.Id, "SalePrice":p}, columns=['Id', 'SalePrice'])
    solution.to_csv("lasso_sol6.csv", index=False)
    
    
mainTest()
