import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier
import sklearn
from sklearn.model_selection import KFold


df = pd.read_csv('D:\\enen\\data\\ASUSdoucment\\python porject\\shaplast\\Newdatathree.csv')
y = df['CLNM']
X = df.drop(['CLNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
df3 = pd.read_csv('D:\\enen\\data\\ASUSdoucment\\python porject\\shaplast\\newdataValth.csv')
y2 = df3['CLNM']
X2 = df3.drop(['CLNM'], axis=1)


params = {'learning_rate': 0.01, 'iterations': 2000, 'depth': 1,
         'subsample': 0.5}

cv= KFold(n_splits=10, shuffle=True)



for i, (train, test) in enumerate(cv.split(X_train, y_train)):
        model = CatBoostClassifier(**params).fit(X_train.iloc[train], y_train.iloc[train])
        predictions = model.predict(X_train.iloc[test])
        actuals = y_train.iloc[test]
        acc = sklearn.metrics.accuracy_score(actuals, predictions)
        print("Accuracy: %.2f%%" % (acc * 100.0))

y_pred = model.predict(X_test)
y_predtrain = model.predict(X_train)
y_dtest = model.predict_proba(X_test)
y_dtrain = model.predict_proba(X_train)


acctest = sklearn.metrics.accuracy_score(y_test, y_pred)
acctrain = sklearn.metrics.accuracy_score(y_train, y_predtrain)


y_d1test = model.predict_proba(X_test)[:, 1]
y_d1train = model.predict_proba(X_train)[:, 1]



print("Accuracyyanzheng: %.2f%%" % (acctest * 100.0))
print("Accuracyxunlian: %.2f%%" % (acctrain * 100.0))


BAtest = sklearn.metrics.balanced_accuracy_score(y_test, y_pred)
BAtrain = sklearn.metrics.balanced_accuracy_score(y_train, y_predtrain)


print("BAyanzheng: %.2f%%" % (BAtest * 100.0))
print("BAxunlian: %.2f%%" % (BAtrain * 100.0))


Ftest = sklearn.metrics.f1_score(y_test, y_pred)
Ftrain = sklearn.metrics.f1_score(y_train, y_predtrain)


print("Fyanzheng: %.2f%%" % (Ftest * 100.0))
print("Fxxunlian: %.2f%%" % (Ftrain * 100.0))


MCCtest = sklearn.metrics.matthews_corrcoef(y_test, y_pred)
MCCtrain = sklearn.metrics.matthews_corrcoef(y_train, y_predtrain)


print("MCCyanzheng: %.2f%%" % (MCCtest * 100.0))
print("MCCxunlian: %.2f%%" % (MCCtrain * 100.0))


PREtest = sklearn.metrics.precision_score(y_test, y_pred)
PREtrain = sklearn.metrics.precision_score(y_train, y_predtrain)


print("PREyanzheng: %.2f%%" % (PREtest * 100.0))
print("PRExunlian: %.2f%%" % (PREtrain * 100.0))


recalltest = sklearn.metrics.recall_score(y_test, y_pred)
recalltrain = sklearn.metrics.recall_score(y_train, y_predtrain)




print("recallyanzheng: %.2f%%" % (recalltest * 100.0))
print("recallxunlian: %.2f%%" % (recalltrain * 100.0))


r2test = sklearn.metrics.r2_score(y_test, y_pred)
r2train = sklearn.metrics.r2_score(y_train, y_predtrain)


print("r2yanzheng: ", r2test)
print("r2xunlian: ", r2train)


rmsetest = np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred))
rmsetrain = np.sqrt(sklearn.metrics.mean_squared_error(y_train, y_predtrain))


print("rmseyanzheng: ", rmsetest)
print("rmsexunlian: ", rmsetrain)


AUCtest = sklearn.metrics.roc_auc_score(y_test, y_d1test)
AUCtrain = sklearn.metrics.roc_auc_score(y_train, y_d1train)



print("AUCyanzheng: %.2f%%" % (AUCtest * 100.0))
print("AUCxunlian: %.2f%%" % (AUCtrain * 100.0))


df1 = pd.DataFrame(y_dtest)
df2 = pd.DataFrame(y_dtrain)

pretest = pd.DataFrame(y_pred)
predtrain = pd.DataFrame(y_predtrain)

df2.to_csv("D:\\enen\\data\\ASUSdoucment\\python porject\\shaplast\\predtraincat10.csv", header=True, index=None)
