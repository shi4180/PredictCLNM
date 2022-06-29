from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_csv('D:\\ASUSdoucment\\python porject\\shaplast\\datatraintest.csv')
y = df['CLNM']
X = df.drop(['CLNM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

X_train.to_csv("D:\\ASUSdoucment\\python porject\\shaplast\\datatrain2.csv", header=True, index=None)
X_test.to_csv("D:\\ASUSdoucment\\python porject\\shaplast\\datatest2.csv", header=True, index=None)
y_train.to_csv("D:\\ASUSdoucment\\python porject\\shaplast\\ytrain2.csv", header=True, index=None)
y_test.to_csv("D:\\ASUSdoucment\\python porject\\shaplast\\ytest2.csv", header=True, index=None)