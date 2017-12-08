import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
# from rmsle import *
import time

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

   
def main():
	# read in the data and build dataframe with pandas
	df_train_set = pd.read_csv('train.csv')
	df_test_set = pd.read_csv('test.csv')
	df_all = pd.concat((df_train_set.loc[:'SaleCondition'],
						df_test_set.loc[:'SaleCondition']), ignore_index=True)
	Y_train = df_train_set['SalePrice']
	Y_train = np.log1p(Y_train)
	# anomaly cases, id have to -1 to get index
	anomalies = np.array([524, 692, 804, 1047, 1299]) - 1
	# drop trivial / irrelevant columns
	columns_to_drop1 = ['Id', 'MSSubClass', 'LotFrontage', 'Alley',
					'LotShape', 'LandContour', 'Utilities',
					'RoofStyle', 'RoofMatl', 'Foundation',
					'Heating', 'Electrical', '1stFlrSF', '2ndFlrSF',
					'KitchenQual', 'GarageCars', 'MiscFeature', 'SalePrice']
	df_tmp = df_all.drop(columns_to_drop1, axis=1)
	df_tmp = df_tmp.drop(anomalies)
	Y_train = Y_train.drop(anomalies)

	#############################################################
	# SOME FEATURE ENGINEERING
	# print df_tmp.columns
	# # LotConfig
	# # MSZoning
	map_Zoning = {'C (all)':0, 'FV':2}
	df_tmp['Zoning_Rating'] = df_tmp.MSZoning.map(map_Zoning).fillna(1).astype(int)
	# # Neighborhood
	map_Neighbor = {'BrDale':0, 'BrkSide':0, 'IDOTRR':0, 'MeadowV':0}
	df_tmp['Neighborhood_Rating'] = df_tmp.Neighborhood.map(map_Neighbor).fillna(1).astype(int)
	# # Condition 1 & 2, dummified and combined together
	dummy_Cond1 = pd.get_dummies(df_tmp['Condition1'])
	dummy_Cond2 = pd.get_dummies(df_tmp['Condition2'])
	dummy_Cond = (dummy_Cond1 | dummy_Cond2).fillna(0).astype(int)
	df_tmp[['Cond_Artery', 'Cond_Feedr', 'Cond_Norm',
			'Cond_RRNn', 'Cond_RRAn', 'Cond_PosN',
			'Cond_PosA', 'Cond_RRNe', 'Cond_RRAe']] = dummy_Cond
	# # BldgType & HouseStyle
	dummy_BldgType = pd.get_dummies(df_tmp['BldgType']).rename(columns={'Twnhs':'TwnhsI'})
	dummy_HouseStyle = pd.get_dummies(df_tmp['HouseStyle'])
	dummy_HouseStyle['1Fam'] = (dummy_HouseStyle['1.5Unf'] | dummy_HouseStyle['2.5Fin'] | dummy_HouseStyle['SLvl'])
	dummy_HouseStyle = dummy_HouseStyle.drop(['1.5Unf', '2.5Fin', 'SLvl'], axis=1)
	dummy_BH = pd.concat([dummy_BldgType.drop(['1Fam'], axis=1), dummy_HouseStyle.drop(['1Fam'], axis=1)], axis=1)
	df_tmp[['BH_2FmCon', 'BH_Duplx', 'BH_TwnhsI',
			'BH_TwnhsE', 'BH_1.5Fin', 'BH_1Story',
			'BH_2.5Unf', 'BH_2Story', 'BH_SFoyer']] = dummy_BH
	# # OverallQual * OverallCond
	df_tmp['Overall_Score'] = df_tmp['OverallQual'] * df_tmp['OverallCond']
	# # YearBuilt, [building age] = [year sold] - [year built]
	df_tmp['Building_Age'] = df_tmp['YrSold'] - df_tmp['YearBuilt']
	# # YearRemodAdd, [remod age] = [year sold] - [year remod]
	df_tmp['Remod_Age'] = df_tmp['YrSold'] - df_tmp['YearRemodAdd']
	# # Exterior 1st & 2nd, dummified and combined together
	Ext_typos = {'Brk Cmn':'BrkComm', 'CmentBd':'CemntBd', 'Wd Shng':'WdShing'}
	dummy_Ext1 = pd.get_dummies(df_tmp['Exterior1st']).rename(columns=Ext_typos)
	dummy_Ext2 = pd.get_dummies(df_tmp['Exterior2nd']).rename(columns=Ext_typos)
	dummy_Ext = (dummy_Ext1 | dummy_Ext2).fillna(0).astype(int)
	df_tmp[['Ext_AsbShng', 'Ext_AsphShn', 'Ext_BrkComm',
			'Ext_BrkFace', 'Ext_CBlock', 'Ext_CemntBd',
			'Ext_HdBoard', 'Ext_ImStcc', 'Ext_MetalSd',
			'Ext_Other', 'Ext_Plywood', 'Ext_Stone',
			'Ext_Stcco', 'Ext_VinylSd', 'Ext_Wd Sdng',
			'Ext_WdShing']] = dummy_Ext
	# # MasVnrType * MasVnrArea
	map_MasVnrType = {'BrkCmn':1, 'BrkFace':1, 'Stone':2}
	MasVnrType_Rating = df_tmp.MasVnrType.map(map_MasVnrType).fillna(0).astype(int)
	df_tmp['MasVnr_Rating'] = MasVnrType_Rating * df_tmp['MasVnrArea'].fillna(0).astype(int)
	# # ExterQual * ExterCond
	map_Rating = {'Ex':10, 'Gd':7, 'TA':5, 'Fa':3, 'Po':2}
	df_tmp['Exter_Rating'] = df_tmp.ExterQual.map(map_Rating).fillna(0).astype(int) \
							* df_tmp.ExterCond.map(map_Rating).fillna(0).astype(int)
	# # (BsmtQual + BsmtExposure) * BsmtCond
	BsmtQual_Rating = df_tmp.BsmtQual.map(map_Rating).fillna(0).astype(int)
	BsmtExposure_Rating = df_tmp.BsmtExposure.map(map_Rating).fillna(0).astype(int)
	BsmtCond_Rating = df_tmp.BsmtCond.map(map_Rating).fillna(0).astype(int)
	df_tmp['Bsmt_Rating'] = (BsmtQual_Rating + BsmtExposure_Rating) * BsmtCond_Rating
	# # BsmtFinType * BsmtFinSF
	map_BsmtFinType = {'GLQ':10, 'ALQ':7, 'Rec':7, 'BLQ':5, 'LwQ':3, 'Unf':2}
	BsmtFinType1_Rating = df_tmp.BsmtFinType1.map(map_BsmtFinType).fillna(0).astype(int)
	BsmtFinType2_Rating = df_tmp.BsmtFinType2.map(map_BsmtFinType).fillna(0).astype(int)
	df_tmp['BsmtFin_Rating'] = BsmtFinType1_Rating * df_tmp['BsmtFinSF1'] \
							+ BsmtFinType2_Rating * df_tmp['BsmtFinSF2']
	# # HeatingQC
	df_tmp['HeatingQC'] = df_tmp.HeatingQC.map(map_Rating).fillna(0).astype(int)
	# # CentralAir
	df_tmp = df_tmp.replace({'CentralAir': {'Y': 1, 'N': 0}})
	# # Functional
	map_Deduction = {'Typ':10, 'Min1':7, 'Min2':7, 'Mod':5, 'Maj1':3, 'Maj2':3, 'Sev':2, 'Sal':0}
	df_tmp['Functional'] = df_tmp.Functional.map(map_Deduction).fillna(0).astype(int)
	# # Fireplaces * FireplaceQu
	df_tmp['Fireplace_Rating'] = df_tmp['Fireplaces'] * df_tmp.FireplaceQu.map(map_Rating).fillna(0).astype(int)
	# # GarageArea * GarageFinish * GarageQual * GarageCond
	map_GarageFin = {'Fin':10, 'RFn':7, 'Unf':5}
	df_tmp['Garage_Rating'] = df_tmp['GarageArea'] * \
							df_tmp.GarageFinish.map(map_GarageFin).fillna(0).astype(int) * \
							df_tmp.GarageQual.map(map_Rating).fillna(0).astype(int) * \
							df_tmp.GarageCond.map(map_Rating).fillna(0).astype(int) / 1000
	# # PavedDrive
	df_tmp = df_tmp.replace({'PavedDrive': {'Y':2, 'P':1, 'N':0}})
	# # PoolArea * PoolQC
	df_tmp['Pool_Rating'] = df_tmp['PoolArea'] * df_tmp.PoolQC.map(map_Rating).fillna(0).astype(int)
	# # Fence
	map_Fence = {'GdPrv':10, 'MnPrv':7, 'GdWo':5, 'MnWw':3, 'NA':0}
	df_tmp['Fence'] = df_tmp.Fence.map(map_Fence).fillna(0).astype(int)
	# # MoSold
	dummy_MoSold = pd.get_dummies(df_tmp['MoSold'])
	df_tmp = pd.concat([df_tmp, dummy_MoSold], axis=1)
	# # YrSold
	dummy_YrSold = pd.get_dummies(df_tmp['YrSold'])
	df_tmp = pd.concat([df_tmp, dummy_YrSold], axis=1)
	# # SaleType
	df_tmp = df_tmp.replace({'SaleType': {'WD':'Deed', 'CWD':'Deed', 'VWD':'Deed',
										'ConLw':'Con', 'ConLI':'Con', 'ConLD':'Con'}})
	# END OF FEATURE ENGINEERING
	#############################################################
	
	# drop redundant columns after feature engineering
	columns_to_drop2 = ['MSZoning', 'Neighborhood', 'Condition1',
						'Condition2', 'BldgType', 'HouseStyle',
						'OverallQual', 'OverallCond', 'YearBuilt',
						'MasVnrType', 'MasVnrArea', 'ExterQual',
						'ExterCond', 'BsmtQual', 'BsmtCond',
						'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
						'BsmtFinType2', 'BsmtFinSF2', 'Fireplaces',
						'FireplaceQu', 'GarageType', 'GarageYrBlt',
						'GarageFinish', 'GarageArea', 'GarageQual',
						'GarageCond', 'PoolArea', 'PoolQC',
						'MoSold', 'YrSold', 'YearRemodAdd', 'SaleCondition'];
	df_tmp = df_tmp.drop(columns=columns_to_drop2)

	# # get dummies
	df_tmp = pd.get_dummies(df_tmp)

	# split train / test sets
	TRAIN_SIZE = df_train_set.shape[0] - len(anomalies)
	X_train = df_tmp[:TRAIN_SIZE]
	X_test = df_tmp[TRAIN_SIZE:]
# 	print X_train.shape, X_test.shape
	df_tmp.to_csv(r'df.csv', sep=',')
	X_train.to_csv(r'XTRAIN.csv', sep=',')
	X_test.to_csv(r'XTEST.csv', sep=',')
	# lasso model
	model_lasso = Lasso(alpha=5e-4, max_iter=1e5).fit(X_train, Y_train)
	p = np.expm1(model_lasso.predict(X_test))

# 	print rmsle(p, np.expm1(Y_train))
	# plt.scatter(p, np.expm1(Y_train))
	# plt.plot([min(p),max(p)], [min(p),max(p)], c="red")
	# plt.show()

	
main()
