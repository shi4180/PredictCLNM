import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
df = pd.read_csv('D:\\ASUSdoucment\\python porject\\shaplast\\Newdatathree.csv')
y = df['CLNM']
X = df.drop(['CLNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
df3 = pd.read_csv('D:\\ASUSdoucment\\python porject\\shaplast\\newdataValth.csv')
y2 = df3['CLNM']
X2 = df3.drop(['CLNM'], axis=1)

classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators= 2000, max_depth= 4, max_leaf_nodes= 3),
    "ANN": MLPClassifier(
        alpha= 0.00001, learning_rate_init= 0.01, power_t= 0.01, max_iter= 110, tol= 0.001,
        momentum= 0.9, beta_1= 0.9, beta_2= 0.999, epsilon= 1e-08, n_iter_no_change= 10, validation_fraction= 0.2,
        hidden_layer_sizes= 200),
    "SVM": SVC(C= 1, kernel='linear',gamma= 0.001, tol=0.0001, cache_size= 200),
    "Decision Tree": DecisionTreeClassifier(max_depth= 4, max_leaf_nodes=4),
    "XGBoost": XGBClassifier(eta= 0.01, n_estimators= 2000, gamma= 5, max_depth= 2, min_child_weight= 1,
                colsample_bytree= 1, colsample_bylevel= 1, colsample_bynode= 1, subsample= 0.5,
                reg_lambda= 1, reg_alpha= 0, seed= 33),

}
cv = KFold(n_splits=10, shuffle=True)
fig, ax_roc = plt.subplots(figsize=(11, 5))
for name, model in classifiers.items():

    for i, (train, test) in enumerate(cv.split(X_train, y_train)):
        model.fit(X_train.iloc[train], y_train.iloc[train])
    plot_roc_curve(model, X2, y2, ax=ax_roc, name=name)

ax_roc.set_title('Receiver Operating Characteristic (ROC) curves')


ax_roc.grid(linestyle='--')


plt.legend()
plt.show()

