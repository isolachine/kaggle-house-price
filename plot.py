import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# from sklearn import linear_model
import numpy as np
# from sklearn.cross_validation import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.metrics import make_scorer, mean_squared_error
# from sklearn.linear_model import Lasso
# import math
# from numpy import mean

def main():
    train_set = pd.read_csv('train.csv')
    
    var = 'Neighborhood'
    varY = 'SalePrice'
    
    
#     print(np.sqrt(train_set[var].describe()))
#     plt.figure(1)
#     sns.distplot(train_set[var])
    plt.figure(1)
#     sns.boxplot(np.log1p(train_set[var]), train_set[varY])
    plt.figure(2)
#     sns.boxplot(np.sqrt(train_set[var]), train_set[varY])
    plt.figure(3)
#     plt.scatter(train_set[var], train_set[varY])
    sns.boxplot(train_set[var], train_set[varY])
    plt.show()
    
    
main()