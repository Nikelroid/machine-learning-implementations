import numpy as np
import pandas as pd

def mean_square_error(w, X, y):
  return np.linalg.norm(X@w - y)**2/len(y)

def linear_regression_noreg(X, y):
  return np.linalg.inv(X.T @ X) @ X.T @ y

def regularized_linear_regression(X, y, lambd):
  return np.linalg.inv(X.T @ X + lambd*np.identity((X).shape[1])) @ X.T @ y

def tune_lambda(Xtrain, ytrain, Xval, yval):
  lambdas = [2**i for i in range(-40, 1)]
  w_list = [regularized_linear_regression(Xtrain,ytrain, lmb) for lmb in lambdas]
  mse_list = [[mean_square_error(w_list[i], Xval, yval),lambdas[i]] for i in range(len(lambdas))]
  mse_list = sorted(mse_list, key=lambda x: x[0])
  return mse_list[0][1]
    

def mapping_data(X, p):
    return np.hstack([np.power(X, i) for i in range(1, p + 1)])

