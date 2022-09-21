import pandas as pd
import sys
import os
import sklearn
import numpy as np
import matplotlib as plm
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from keras.losses import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, reciprocal

df=pd.read_csv("D:\\qikan\\tf\\Delete default del.csv",parse_dates=["No"],index_col=[0])
print(df.head())
#print(df.tail())
print(df.shape)#(41757, 7)

X = df[['DEWP','TEMP','PRES','Iws','Is','Ir']]
Y = df[['pm2.5']]

print(X.shape)
print(Y.shape)
print('****************')

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)#,random_state=42
print(Y_test.shape)
print(Y_train.shape)
print(X_test.shape)
print(X_train.shape)
print('****************')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# #线性SVR
# lin_svr = LinearSVR()#random_state=42
# lin_svr.fit(X_train_scaled,Y_train)
#
# #模型测试
# Y_pred = lin_svr.predict(X_train_scaled)
# mse = mean_squared_error(Y_train,Y_pred)
# mae = mean_absolute_error(Y_train,Y_pred)
# rmse = np.sqrt(mse)
# print(mse)
# print(mae)
# print(rmse)

#核函数的SVR
param_distribution = {"gamma":reciprocal(0.001,0.1),"C":uniform(1,10)}
rnd_search_cv = RandomizedSearchCV(SVR(),param_distribution,n_iter=10,verbose=2,cv=3)#random_state = 42
rnd_search_cv.fit(X_train_scaled,Y_train)

#得到最优参数
print(rnd_search_cv.best_estimator_)

Y_pred = rnd_search_cv.best_estimator_.predict(X_train_scaled)
mse = mean_squared_error(Y_train,Y_pred)
mae = mean_absolute_error(Y_train,Y_pred)
rmse = np.sqrt(mse)
print(mse)
print(mae)
print(rmse)
print("****************")
y_test_pred = rnd_search_cv.best_estimator_.predict(X_test_scaled)
mse = mean_squared_error(Y_test,y_test_pred)
mae = mean_absolute_error(Y_test,y_test_pred)
rmse = np.sqrt(mse)
print(mse)
print(mae)
print(rmse)

#
plt.plot(Y_test, label = 'Actual PM2.5 concentration')#, color = 'red'
plt.plot(y_test_pred, label = 'Predicted PM2.5 concentration')#, color = 'blue'
plt.title('PM2.5 concentration prediction')
plt.xlabel('Time')
plt.ylabel('PM2.5 concentration')
plt.legend()
plt.show()