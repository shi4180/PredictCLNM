import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
df = pd.read_csv('F:\\python porject\\ShapPredictCLNM6\\data5newTrainTest.csv')
y = df['CLNM']
X = df.drop(['CLNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

other_params = {'eta': 0.02, 'n_estimators': 500, 'gamma': 0, 'max_depth': 2, 'min_child_weight': 1,
                'colsample_bytree': 1, 'colsample_bylevel': 1, 'subsample': 1, 'reg_lambda': 1, 'reg_alpha': 0,
                'seed': 33}
cv_params = {'n_estimators': np.linspace(0, 3000, 10, dtype=int)}
model = xgboost.XGBClassifier(**other_params)
gs = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)
