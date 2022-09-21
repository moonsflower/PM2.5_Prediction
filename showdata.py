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

plt.plot(df['pm2.5'], label = 'Actual PM2.5 concentration')

plt.title('PM2.5 concentration')
plt.xlabel('Time')
plt.ylabel('PM2.5 concentration')
plt.legend()
plt.show()