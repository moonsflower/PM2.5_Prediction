import math
import os
import numpy as np
import pandas as pd
from keras import Model
from keras.layers import GRU, Conv1D, BatchNormalization, Activation
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


df_for_training=df[:-8351]
df_for_testing=df[-8351:]
print(df_for_training.shape)#(33406, 7)
print(df_for_testing.shape)#(8351, 7)

scaler = MinMaxScaler(feature_range=(0,1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled=scaler.transform(df_for_testing)
print(df_for_training_scaled)
#
#N_past是我们在预测下一个目标值时将在过去查看的步骤数。
#
#这里使用30，意味着将使用过去的30个值(包括目标列在内的所有特性)来预测第31个目标值。

def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)

trainX,trainY=createXY(df_for_training_scaled,30)
testX,testY=createXY(df_for_testing_scaled,30)

print("trainX Shape-- ",trainX.shape)#(33376, 30, 7)
print("trainY Shape-- ",trainY.shape)#(33376,)

print("testX Shape-- ",testX.shape)#(8321, 30, 7)
print("testY Shape-- ",testY.shape)#(8321,)

# print("trainX[0]-- \n",trainX[0])
# print("trainY[0]-- ",trainY[0])
class ResnetBlock(Model):

    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv1D(filters, 3, strides=strides, padding='same', use_bias=False,kernel_regularizer=tf.keras.regularizers.l2())
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.d1 = Dropout(0.2)

        self.c2 = Conv1D(filters, 3, strides=1, padding='same', use_bias=False,kernel_regularizer=tf.keras.regularizers.l2())
        self.b2 = BatchNormalization()

        # residual_path为True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
        if residual_path:
            self.down_c1 = Conv1D(filters, 1, strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs  # residual等于输入值本身，即residual=x
        # 将输入通过卷积、BN层、激活层，计算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(y + residual)  # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        return out

class ResNet18(Model):

    def __init__(self, block_list, initial_filters=64):  # block_list表示每个block有几个卷积层
        super(ResNet18, self).__init__()
        self.num_blocks = len(block_list)  # 共有几个block
        self.block_list = block_list
        self.out_filters = initial_filters
        self.c1 = Conv1D(self.out_filters, 3, strides=1, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = tf.keras.models.Sequential()
        # 构建ResNet网络结构
        for block_id in range(len(block_list)):  # 第几个resnet block
            for layer_id in range(block_list[block_id]):  # 第几个卷积层

                if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                    block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters, residual_path=False)
                self.blocks.add(block)  # 将构建好的block加入resnet
            self.out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍
        self.p1 = tf.keras.layers.GlobalAveragePooling1D()
        self.f1 = tf.keras.layers.Dense(1)#, kernel_regularizer=tf.keras.regularizers.l2()
        self.g1 = GRU(50, return_sequences=True)
        self.d1 = Dropout(0.3)
        self.g2 = GRU(50, return_sequences=False)

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)

        x = self.g1(x)
        x = self.d1(x)
        x = self.g2(x)
        x = self.d1(x)
        y = self.f1(x)
        return y


model = ResNet18([2, 2, 2, 2])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')

checkpoint_save_path = "./checkpoint/Resnet+GRU50.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')
grid_search = model.fit(trainX,trainY,batch_size=64, epochs=1, validation_freq=1,validation_data=(testX,testY),callbacks=[cp_callback])

model.summary()

prediction=model.predict(testX)
print("prediction\n", prediction)
print("\nPrediction Shape-",prediction.shape)

prediction_copies_array = np.repeat(prediction,7, axis=-1)
#prediction_copies_array.shape
pred=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),7)))[:,0]
original_copies_array = np.repeat(testY,7, axis=-1)
original=scaler.inverse_transform(np.reshape(original_copies_array,(len(testY),7)))[:,0]

print("Pred Values-- " ,pred)
print("\nOriginal Values-- " ,original)

plt.plot(original, label = 'Actual PM2.5 concentration')#, color = 'red'
plt.plot(pred, label = 'Predicted PM2.5 concentration')#, color = 'blue'
plt.title('PM2.5 concentration prediction')
plt.xlabel('Time')
plt.ylabel('PM2.5 concentration')
plt.legend()
plt.show()

loss = grid_search.history['loss']
val_loss = grid_search.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

##########evaluate##############
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(pred, original)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(pred, original))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(pred, original)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)