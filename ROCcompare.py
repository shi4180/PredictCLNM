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
df = pd.read_csv('D:\\enen\\data\\ASUSdoucment\\python porject\\shaplast\\Newdatathree.csv')
y = df['CLNM']
X = df.drop(['CLNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
df3 = pd.read_csv('D:\\enen\\data\\ASUSdoucment\\python porject\\shaplast\\newdataValth.csv')
y2 = df3['CLNM']
X2 = df3.drop(['CLNM'], axis=1)
paramsXGB = {'eta': 0.01, 'n_estimators': 2700, 'gamma': 5, 'max_depth': 2, 'min_child_weight': 1,
                'colsample_bytree': 1, 'colsample_bylevel': 1, 'colsample_bynode': 1, 'subsample': 0.5,
                'reg_lambda': 1, 'reg_alpha': 0, 'seed': 33}
cv= KFold(n_splits=10, shuffle=True)
cvXGB= KFold(n_splits=10,  random_state=0, shuffle=True)
for i, (train, test) in enumerate(cvXGB.split(X_train, y_train)):
        modelXGB = XGBClassifier(**paramsXGB).fit(X_train.iloc[train], y_train.iloc[train])
y_predXGB = modelXGB.predict(X_train)
y_dtestXGB = modelXGB.predict_proba(X_train)
y_d1test2XGB = modelXGB.predict_proba(X_train)[:, 1]
y2_d1test2XGB = pd.DataFrame(y_d1test2XGB)

from sklearn.ensemble import RandomForestClassifier
paramsRF = {'n_estimators': 2000, 'max_depth': 4, 'max_leaf_nodes': 3}
for i, (train, test) in enumerate(cv.split(X_train, y_train)):
        modelRF = RandomForestClassifier(**paramsRF).fit(X_train.iloc[train], y_train.iloc[train])
y_predRF = modelRF.predict(X_train)
y_dtestRF = modelRF.predict_proba(X_train)
y_d1test2RF = modelRF.predict_proba(X_train)[:, 1]
y2_d1test2RF = pd.DataFrame(y_d1test2RF)


from sklearn.neural_network import MLPClassifier
paramsANN = {'alpha': 0.00001, 'learning_rate_init': 0.01, 'power_t':0.01, 'max_iter': 110, 'tol':0.001,
          'momentum':0.9, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':1e-08, 'n_iter_no_change':10, 'validation_fraction':0.2,
          'hidden_layer_sizes':200}
for i, (train, test) in enumerate(cv.split(X_train, y_train)):
        modelANN = MLPClassifier(**paramsANN, activation="tanh", solver="adam",
        learning_rate="constant",
      shuffle=True,
      verbose=False,
    ).fit(X_train.iloc[train], y_train.iloc[train])
y_predANN = modelANN.predict(X_train)
y_dtestANN = modelANN.predict_proba(X_train)
y_d1test2ANN = modelANN.predict_proba(X_train)[:, 1]
y2_d1test2ANN = pd.DataFrame(y_d1test2ANN)

from sklearn.svm import SVC
paramsSVM = {'C': 1, 'kernel':'linear','gamma': 0.001}
for i, (train, test) in enumerate(cv.split(X_train, y_train)):
        modelSVM = SVC(**paramsSVM, probability=True).fit(X_train.iloc[train], y_train.iloc[train])
y_predSVM = modelSVM.predict(X_train)
y_dtestSVM = modelSVM.predict_proba(X_train)
y_d1test2SVM = modelSVM.predict_proba(X_train)[:, 1]
y2_d1test2SVM = pd.DataFrame(y_d1test2SVM)

from sklearn.tree import DecisionTreeClassifier
paramsTREE = {'max_depth': 4, 'max_leaf_nodes':4}
for i, (train, test) in enumerate(cv.split(X_train, y_train)):
        modelTREE = DecisionTreeClassifier(**paramsTREE).fit(X_train.iloc[train], y_train.iloc[train])
y_predTREE = modelTREE.predict(X_train)
y_dtestTREE = modelTREE.predict_proba(X_train)
y_d1test2TREE = modelTREE.predict_proba(X_train)[:, 1]
y2_d1test2TREE = pd.DataFrame(y_d1test2TREE)
from lightgbm import LGBMClassifier


lightparams = {'learning_rate': 0.01, 'n_estimators': 2700, 'max_depth': 1, 'min_child_weight': 1,
                'colsample_bytree': 1,  'colsample_bynode': 1, 'subsample': 0.5,
          'reg_lambda': 1, 'reg_alpha': 0}
lightcv= KFold(n_splits=10, shuffle=True)
for i, (train, test) in enumerate(lightcv.split(X_train, y_train)):
        modellight = LGBMClassifier(**lightparams).fit(X_train.iloc[train], y_train.iloc[train])
y_predlight = modellight.predict(X_train)
y_dtestlight = modellight.predict_proba(X_train)
y_d1test2light = modellight.predict_proba(X_train)[:, 1]

from catboost import CatBoostClassifier
catparams = {'learning_rate': 0.01, 'iterations': 2000, 'depth': 1,
         'subsample': 0.5}
catcv= KFold(n_splits=10, shuffle=True)
for i, (train, test) in enumerate(catcv.split(X_train, y_train)):
        catmodel = CatBoostClassifier(**catparams).fit(X_train.iloc[train], y_train.iloc[train])
y_predcat = modellight.predict(X_train)
y_dtestcat = modellight.predict_proba(X_train)
y_d1test2cat = modellight.predict_proba(X_train)[:, 1]
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
fpr_XGB, tpr_XGB, threshold_XGB = roc_curve(y_train, y_d1test2XGB)  ###计算真正率和假正率
roc_auc_XGB = auc(fpr_XGB, tpr_XGB)  ###计算auc的值
fpr_ANN, tpr_ANN, threshold_ANN = roc_curve(y_train, y_d1test2ANN)  ###计算真正率和假正率
roc_auc_ANN = auc(fpr_ANN, tpr_ANN)  ###计算auc的值
fpr_RF, tpr_RF, threshold_RF = roc_curve(y_train, y_d1test2RF)  ###计算真正率和假正率
roc_auc_RF = auc(fpr_RF, tpr_RF)  ###计算auc的值
fpr_SVM, tpr_SVM, threshold_SVM = roc_curve(y_train, y_d1test2SVM)  ###计算真正率和假正率
roc_auc_SVM = auc(fpr_SVM, tpr_SVM)  ###计算auc的值
fpr_TREE, tpr_TREE, threshold_TREE = roc_curve(y_train, y_d1test2TREE)  ###计算真正率和假正率
roc_auc_TREE = auc(fpr_TREE, tpr_TREE)  ###计算auc的值
fpr_GBM, tpr_GBM, threshold_GBM = roc_curve(y_train, y_d1test2light)  ###计算真正率和假正率
roc_auc_GBM = auc(fpr_GBM, tpr_GBM)  ###计算auc的值
fpr_cat, tpr_cat, threshold_cat = roc_curve(y_train, y_d1test2cat)  ###计算真正率和假正率
roc_auc_cat = auc(fpr_cat, tpr_cat)  ###计算auc的值
plt.figure(figsize=(8, 5))
plt.plot(fpr_XGB, tpr_XGB, color='darkorange', ###假正率为横坐标，真正率为纵坐标做曲线
         lw=2, label='XGBoost (AUC = %0.2f)' % roc_auc_XGB, linestyle='-') #linestyle为线条的风格（共五种）,color为线条颜色
plt.plot(fpr_ANN, tpr_ANN, color='green',
         lw=2, label='ANN (AUC = %0.2f)' % roc_auc_ANN, linestyle='--')
plt.plot(fpr_RF, tpr_RF, color='red',
         lw=2, label='Random Forest (AUC = %0.2f)' % roc_auc_RF, linestyle='--')
plt.plot(fpr_SVM, tpr_SVM, color='#800080',
         lw=2, label='SVM (AUC = %0.2f)' % roc_auc_SVM, linestyle=':')
plt.plot(fpr_TREE, tpr_TREE, color='#D2691E',
         lw=2, label='Decision Tree (AUC = %0.2f)' % roc_auc_TREE, linestyle='-.')
plt.plot(fpr_GBM, tpr_GBM, color='#D2691E',
         lw=2, label='LightGBM (AUC = %0.3f)' % roc_auc_GBM, linestyle='-.');
plt.plot(fpr_cat, tpr_cat, color='#D2691E',
         lw=2, label='Lightcat (AUC = %0.3f)' % roc_auc_cat, linestyle='-.');
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.02, 1.05])#横竖增加一点长度 以便更好观察图像
plt.ylim([-0.02, 1.05])
plt.legend(loc="lower right")
plt.show()
plt.figure(figsize=(8, 5))
plt.plot(fpr_XGB, tpr_XGB, color='#64B01C', ###假正率为横坐标，真正率为纵坐标做曲线
         lw=2, label='XGBoost (AUC = %0.2f)' % roc_auc_XGB, linestyle='-') #linestyle为线条的风格（共五种）,color为线条颜色
plt.plot(fpr_ANN, tpr_ANN, color='#4D6FFF',
         lw=2, label='ANN (AUC = %0.2f)' % roc_auc_ANN, linestyle='-')
plt.plot(fpr_RF, tpr_RF, color='red',
         lw=2, label='Random Forest (AUC = %0.2f)' % roc_auc_RF, linestyle='-')
plt.plot(fpr_SVM, tpr_SVM, color='#9DFC42',
         lw=2, label='SVM (AUC = %0.2f)' % roc_auc_SVM, linestyle='-')
plt.plot(fpr_TREE, tpr_TREE, color='#FC5A28',
         lw=2, label='Decision Tree (AUC = %0.2f)' % roc_auc_TREE, linestyle='-')
plt.plot(fpr_GBM, tpr_GBM, color='#B3664F',
         lw=2, label='LightGBM (AUC = %0.2f)' % roc_auc_GBM, linestyle='-');
plt.plot(fpr_cat, tpr_cat, color='#57C4DE',
         lw=2, label='CatBoost (AUC = %0.2f)' % roc_auc_cat, linestyle='-');
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='-')
plt.xlim([-0.02, 1.05])#横竖增加一点长度 以便更好观察图像
plt.ylim([-0.02, 1.05])
plt.legend(loc="lower right")
plt.show()