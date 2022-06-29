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

dfcat = pd.read_csv('D:\\ASUSdoucment\\python porject\\shaplast\\Newdatathreetiao2cat.csv')
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
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=False)


explainer = shap.TreeExplainer(model)
ndf =df.sort_values(by="CLNM")
print(ndf)

X2 = ndf[0:671].drop(['CLNM'], axis=1)
X1 = ndf[672:1121].drop(['CLNM'], axis=1)
shap_values = explainer.shap_values(X)
shap.monitoring_plot(0, shap_values, X)
shap.monitoring_plot(1, shap_values, X)
shap.monitoring_plot(10, shap_values, X)
shap.monitoring_plot(6, shap_values, X)
shap.monitoring_plot(13, shap_values, X)
shap.monitoring_plot(2, shap_values, X)
shap.embedding_plot(0, shap_values, feature_names=X.columns)
shap.embedding_plot(1, shap_values, feature_names=X.columns)
shap.embedding_plot(10, shap_values, feature_names=X.columns)
shap.embedding_plot(6, shap_values, feature_names=X.columns)
shap.embedding_plot(13, shap_values, feature_names=X.columns)
shap.embedding_plot(2, shap_values, feature_names=X.columns)
shap_values1 = explainer.shap_values(X1[1:449])
shap_values2 = explainer.shap_values(X2[1:449])
shap.summary_plot(shap_values, X, plot_type="bar", color="Blue")
shap.summary_plot(shap_values, X, plot_type="dot")
shap.summary_plot([shap_values1, shap_values2], X, plot_type="bar", class_names=["non-metastasis","metastasis"])


shap_interaction_values = explainer.shap_interaction_values(X)
shap.summary_plot(shap_interaction_values, X)
shap.dependence_plot("Capsular Invasion ", shap_values, Xcat, interaction_index="Diameter")
shap.dependence_plot("Capsular Invasion ", shap_values, Xcat, interaction_index="ICIVP")
shap.dependence_plot("Capsular Invasion ", shap_values, Xcat, interaction_index="Calcification")
shap.dependence_plot("Capsular Invasion ", shap_values, Xcat, interaction_index="Sex")
shap.dependence_plot("Capsular Invasion ", shap_values, Xcat, interaction_index="Age")


shap.dependence_plot("Sex", shap_values, Xcat, interaction_index=None)
shap.dependence_plot("ICIVP", shap_values, Xcat, interaction_index=None)
shap.dependence_plot("Calcification", shap_values, Xcat, interaction_index=None)
shap.dependence_plot("Capsular Invasion ", shap_values, Xcat, interaction_index=None)
shap.dependence_plot("Diameter", shap_values, Xcat, interaction_index=None)
shap.dependence_plot("Age", shap_values, Xcat, interaction_index=None)





