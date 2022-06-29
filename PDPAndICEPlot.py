import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import shap
import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate
from sklearn.inspection import plot_partial_dependence
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
df = pd.read_csv('D:\\ASUSdoucment\\python porject\\shaplast\\Newdatathree2.csv')
y = df['CLNM']
X = df.drop(['CLNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

dfcat = pd.read_csv('D:\\ASUSdoucment\\python porject\\shaplast\\Newdatathreecat.csv')
ycat = dfcat['CLNM']
Xcat = dfcat.drop(['CLNM'], axis=1)
Xcat=pd.DataFrame(Xcat)
ycat=pd.DataFrame(ycat)
X_traincat, X_testcat, y_traincat, y_testcat = train_test_split(Xcat, ycat, test_size=0.2, random_state=2)

params = {'eta': 0.01, 'n_estimators': 2700, 'gamma': 5, 'max_depth': 2, 'min_child_weight': 1,
                'colsample_bytree': 1, 'colsample_bylevel': 1,  'subsample': 0.5,
                'reg_lambda': 1, 'reg_alpha': 0, 'seed': 33}
model = XGBClassifier(**params)
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="logloss", eval_set=eval_set, verbose=False)
features = ['Capsular Invasion ',  'Diameter', 'Calcification', 'ICIVP', 'Sex', 'Age']
display = plot_partial_dependence(model, X, features, kind="both", subsample=50,
       n_jobs=-1, grid_resolution=20, random_state=0)
display.figure_.subplots_adjust(wspace=0.4, hspace=0.3)




