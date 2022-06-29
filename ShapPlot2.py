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
df = pd.read_csv('D:\\ASUSdoucment\\python porject\\shaplast\\Newdatathree.csv')
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
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.decision_plot(explainer.expected_value, shap_values[730, :], Xcat.iloc[730, :], link='logit')
shap.bar_plot(shap_values[730, :], feature_names=Xcat.columns)
shap.decision_plot(explainer.expected_value, shap_values, Xcat, link='logit')
shap.decision_plot(explainer.expected_value, shap_values, Xcat, feature_order="hclust")
shap.plots.force(explainer.expected_value, shap_values[730, :], Xcat.iloc[730, :], link="logit", matplotlib=True)
