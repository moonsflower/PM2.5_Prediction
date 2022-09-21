import math

import tensorflow as tf
import os
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import pandas as pd
from keras.losses import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv1D,GlobalMaxPool1D,MaxPooling1D
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

info = pd.read_csv('D:\\qikan\\tf\\Delete default.csv')
print(info)
x_train = info[['DEWP','TEMP','PRES','Iws','Is','Ir']]#这里没用cbwd
y_train = info['pm2.5']
print(x_train.shape)
print(y_train.shape)
# tf.shape(np.expand_dims(x_train, axis=2))
# tf.shape(np.expand_dims(y_train, axis=1))

x_train1 = tf.cast(x_train,tf.float32)
y_train1 = tf.cast(y_train,tf.float32)
print(str(x_train.shape))
print(str(y_train.shape))

x_train = x_train1[:-8351]
y_train = y_train1[:-8351]
x_test = x_train1[-8351:]
y_test = y_train1[-8351:]

x_train = tf.expand_dims(x_train, -1)
x_test = tf.expand_dims(x_test, -1)
print(x_train.shape)
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

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


model = ResNet18([2, 2, 2, 2])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')

checkpoint_save_path = "./checkpoint/Resnet.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')
history = model.fit(x_train, y_train, batch_size=64, epochs=1, validation_data=(x_test,y_test), validation_freq=1,callbacks=[cp_callback])#
model.summary()

prediction=model.predict(x_test)
print("prediction\n", prediction)
print("\nPrediction Shape-",prediction.shape)

plt.plot(y_test, label = 'Actual PM2.5 concentration')#, color = 'red'
plt.plot(prediction, label = 'Predicted PM2.5 concentration')#, color = 'blue'
plt.title('PM2.5 concentration prediction')
plt.xlabel('Time')
plt.ylabel('PM2.5 concentration')
plt.legend()
plt.show()


loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

print(prediction.shape)
print(y_test.shape)
prediction = prediction.reshape(8351)


##########evaluate##############
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(prediction, y_test)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(prediction, y_test))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(prediction, y_test)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)