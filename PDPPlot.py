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
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots
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
pdp_goals = pdp.pdp_isolate(model=model, dataset=X, model_features=Xcat.columns, feature='Sex')
pdp.pdp_plot(pdp_goals, 'Sex')
plt.show()
features_to_plot = ['Capsular Invasion ', 'Shape']
inter1  =  pdp.pdp_interact(model=model, dataset=X, model_features=Xcat.columns, features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour', x_quantile=True,
                      plot_pdp=True)
plt.show()