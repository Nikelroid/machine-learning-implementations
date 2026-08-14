import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import precision_recall_fscore_support as prfs

from boosting import AdaBoost

import torchvision.datasets as datasets
import torchvision.transforms as transforms

def train_model(X_train, t_train, max_depth = 10, min_samples_leaf = 10, criterion = 'gini', 
                seed = 60, model_type = 'DT', n_estimators = 50, max_samples = 0.5, max_features = 100):
 
  if model_type == 'DT':
    model = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = min_samples_leaf,
    criterion = criterion, random_state = seed)
  
  elif model_type == 'RF':
    model = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators, min_samples_leaf=min_samples_leaf,
    criterion = criterion, max_samples = max_samples, max_features = max_features, random_state = seed)
    
  elif model_type == 'AdaBoost':
    model = AdaBoost(n_estimators = n_estimators,max_depth = max_depth)
  
  model = model.fit(X_train, t_train)
  return model


def prediction_model(X_test, model, print_flag = False):
  
  y_pred = model.predict(X_test)
  
  if print_flag:
    print("Prediction is: ", y_pred)

  return y_pred

def test_model(X_test, t_test, model, print_flag = False):

  y_pred = model.predict(X_test)
  acc = model.score(X_test, t_test)
  
  if print_flag:
    print("Accuracy is: ", acc)

  return acc

def plot_acc(acc_tr, acc_te, par_val, par_name, model_type, ymin = 0.75):
  
  fig, ax = plt.subplots()

  ax.plot(par_val, acc_tr, color = 'red', label = 'train')
  ax.plot(par_val, acc_te, color = 'blue', label = 'test')

  ax.legend()
  ax.set_ylim(ymin, 1.0)
  ax.set(xlabel = par_name, ylabel = 'Accuracy')
  ax.set_title(('Accuracy for different values of ' + par_name))

  fig.savefig('Accuracy_' + par_name + '_' + model_type + '.png')

  plt.show()
    
def get_acc(X_tr, t_tr, X_te, t_te, par_val, par_name, model_type):
   
   acc_tr = []
   acc_te = []
   
   for i in par_val:

     if par_name == 'max_depth':
       model = train_model(X_tr, t_tr, model_type = model_type, max_depth = i, min_samples_leaf = 1) 

     elif par_name == 'min_samples_leaf':
       model = train_model(X_tr, t_tr, model_type = model_type, min_samples_leaf = i, max_depth = 10)

     elif par_name == 'n_estimators':
       model = train_model(X_tr, t_tr, model_type = model_type, n_estimators = i)

     elif par_name == 'max_features':
       model = train_model(X_tr, t_tr, model_type = model_type, max_features = i)

     elif par_name == 'max_samples':
       model = train_model(X_tr, t_tr, model_type = model_type, max_samples = i)

     ac_tr = test_model(X_tr, t_tr, model)
     ac_te = test_model(X_te, t_te, model)

     acc_tr.append(ac_tr)
     acc_te.append(ac_te)

   return np.array(acc_tr), np.array(acc_te)

def check_behavior(X_tr, X_te, y_tr, y_te, vals, name, model_type, ylim = 0.75):
  acc_tr, acc_te = get_acc(X_tr, y_tr, X_te, y_te, par_val=vals, par_name=name, model_type = model_type)
  plot_acc(acc_tr, acc_te, vals, name, model_type, ylim)

