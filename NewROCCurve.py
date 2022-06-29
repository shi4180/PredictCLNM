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
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RepeatedKFold
df = pd.read_csv('D:\\ASUSdoucment\\python porject\\shaplast\\Newdatathree.csv')
y = df['CLNM']
X = df.drop(['CLNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
dtrain = xgboost.DMatrix(X_train, label=y_train)
dtest = xgboost.DMatrix(X_test, label=y_test)
params = {'eta': 0.01, 'n_estimators': 2700, 'gamma': 5, 'max_depth': 2, 'min_child_weight': 1,
                'colsample_bytree': 1, 'colsample_bylevel': 1,  'subsample': 0.5,
                'reg_lambda': 1, 'reg_alpha': 0, 'seed': 33}
cv=KFold(n_splits=10, random_state=0, shuffle=True)



for i, (train, test) in enumerate(cv.split(X_train, y_train)):
        model = XGBClassifier(**params).fit(X_train.iloc[train], y_train.iloc[train])
        predictions = model.predict(X_train.iloc[test])
        actuals = y_train.iloc[test]
        acc = sklearn.metrics.accuracy_score(actuals, predictions)
        print("Accuracy: %.2f%%" % (acc * 100.0))
xgboosttrain = plot_roc_curve(model, X_train, y_train, name="XGBClassfier in the training data set")
xgboosttest = plot_roc_curve(model, X_test, y_test, ax=xgboosttrain.ax_, name="XGBClassfier in the test data set")




plt.show()

