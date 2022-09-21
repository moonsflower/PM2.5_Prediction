import math
import os
import numpy as np
import pandas as pd
from keras import layers
from keras.layers import GRU, Bidirectional
from keras.losses import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
df=pd.read_csv("D:\\qikan\\tf\\Delete default del.csv",parse_dates=["No"],index_col=[0])
print(df.head())
#print(df.tail())
print(df.shape)#(41757, 7)

fig = plt.figure()
fig.tight_layout()
plt.subplot(3,1,1)
plt.plot(df['pm2.5']/20, label = ' PM2.5 concentration')
plt.plot(df['DEWP'], label = 'Dew Point')
plt.title('(a) PM2.5-DEWP')
plt.ylabel('')
plt.legend()

plt.subplot(3,1,2)
plt.plot(df['pm2.5']/20, label = ' PM2.5 concentration')
plt.plot(df['TEMP'], label = 'Temperature')
plt.title('(b) PM2.5-TEMP')
plt.ylabel('')
plt.legend()
plt.tight_layout()

plt.subplot(3,1,3)
plt.plot(df['pm2.5']/20, label = ' PM2.5 concentration')
plt.plot(df['Iws']/20, label = 'Cumulated wind speed')
plt.title('(c) PM2.5-Iws')
plt.xlabel('Time')
plt.ylabel('')
plt.legend()
plt.tight_layout()
plt.show()
#
# fig = plt.figure()
# fig.tight_layout()
# plt.subplot(3,1,1)
# plt.plot(df['Iws'], label = 'Cumulated wind speed')
# plt.plot(df['DEWP']*20, label = 'Dew Point')
# plt.title('(a) Iws-DEWP')
# plt.ylabel('')
# plt.legend()
#
# plt.subplot(3,1,2)
# plt.plot(df['Iws'], label = 'Cumulated wind speed')
# plt.plot(df['TEMP']*20, label = 'Temperature')
# plt.title('(b) Iws-TEMP')
# plt.ylabel('')
# plt.legend()
# plt.tight_layout()
#
# plt.subplot(3,1,3)
# plt.plot(df['Iws'], label = 'Cumulated wind speed')
# plt.plot((df['PRES']-1000)*20, label = 'Pressure')
# plt.title('(c) Iws-PRES')
# plt.xlabel('Time')
# plt.ylabel('')
# plt.legend()
# plt.tight_layout()
# plt.show()
