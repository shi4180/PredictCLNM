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
from sklearn.model_selection import KFold
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
cv= KFold(n_splits=10, random_state=0, shuffle=True)



for i, (train, test) in enumerate(cv.split(X_train, y_train)):
        model = XGBClassifier(**params).fit(X_train.iloc[train], y_train.iloc[train])
explainer = shap.TreeExplainer(model)
expected_value = explainer.expected_value

select = range(100)
features = X.iloc[15:35]
features_display = Xcat.loc[features.index]

shap_values = explainer.shap_values(features)


shap.decision_plot(expected_value, shap_values, features_display, link='logit')
y_pred = (shap_values.sum(1) + expected_value) > 0
misclassified = y_pred != y[15:35]
shap.decision_plot(expected_value, shap_values, features_display, link='logit', highlight=misclassified)
shap.decision_plot(expected_value, shap_values[misclassified], features_display[misclassified],
                   link='logit', highlight=0)
shap.force_plot(expected_value, shap_values[misclassified], features_display[misclassified],
                link='identity', matplotlib=True)
shap.force_plot(expected_value, shap_values[misclassified], features_display[misclassified],
                link='logit',  matplotlib=True)