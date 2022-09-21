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





class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = Conv1D(filters=6, kernel_size=5, padding='same')  # 卷积层C
        self.b1 = BatchNormalization()     # BN层B
        self.a1 = Activation('relu')  # 激活层A
        self.p1 = MaxPooling1D(pool_size=2, strides=2, padding='same')  # 池化层P
        self.d1 = Dropout(0.2)  # dropout层D

        self.flatten = Flatten()
        self.f1 = Dense(512, activation='relu')
        self.f1 = Dense(128, activation='relu')
        self.d2 = Dropout(0.2)
        self.f2 = Dense(1)

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y


model = Baseline()

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')

checkpoint_save_path = "./checkpoint/CNN.ckpt"

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