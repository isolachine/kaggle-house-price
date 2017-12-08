from sklearn.model_selection import GridSearchCV
# Adding needed libraries and reading data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle

# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')

train_set = pd.read_csv('data/train.csv')
test_set = pd.read_csv('data/test.csv')

outliers_id = np.array([1299,524,633,1325,31,463,969])
outliers_id = outliers_id - 1 # id starts with 1, index starts with 0
train_set = train_set.drop(train_set.index[outliers_id])
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
X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
X_train = X[:train_set.shape[0]]
X_test = X[train_set.shape[0]:]

y = np.log1p(train_set.SalePrice)
# Grid Search for Algorithm Tuning
# prepare a range of alpha values to test
# alphas = np.array([0.00104436636486,5e-4])
paras = {'learning_rate':[0.1,0.2,0.3],'n_estimators':[1000,2000,3000],'max_depth':[2,3,4] }
# create and fit a ridge regression model, testing each alpha
# model = ElasticNet()
# model = Lasso()
model = ensemble.GradientBoostingRegressor()
grid = GridSearchCV(model,paras)
grid.fit(X_train,  y)
print(grid)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_.learning_rate, grid.best_estimator_.n_estimators,grid.best_estimator_.max_depth)