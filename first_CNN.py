# -*- coding: utf-8 -*-
from __future__ import print_function

from  keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as k
# np.random.seed(1337)
from keras.preprocessing.image import ImageDataGenerator

#输入图像维度，
img_rows,img_cols=28,28


# keras中的mnist数据集已经被划分成了60,000个训练集，10,000个测试集的形式
(x_train,y_train),(x_test,y_test)=mnist.load_data()
number=10000
# 后端使用tensorflow时，即tf模式下，
# 会将100张RGB三通道的16*32彩色图表示为(100,16,32,3)，
# 第一个维度是样本维，表示样本的数目，
# 第二和第三个维度是高和宽，
# 最后一个维度是通道维，表示颜色通道数
x_train = x_train[0:number]
y_train = y_train[0:number]
x_train=x_train.reshape(number,img_rows,img_cols,1)
# x_train=x_train.reshape(number,1,img_rows,img_cols)
x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,1)

#将数据进行转化
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

#打印相关信息
print('x_train shape',x_train.shape)
print(x_train.shape[0],'train sample')
print(x_test.shape[0],'test sample')

#将类别向量(从0到nb_classes的整数向量)映射为二值类型矩阵
#相当于将向量进行one_hot编码
y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)

x_train = x_train
x_test = x_test
# normalization非常关键
x_train = x_train / 255
x_test = x_test / 255

model=Sequential()
#卷积
model.add(Conv2D(64,3,3,input_shape=(28,28,1),name='conv1_1'))#获得的新的图像是26*26，有25个。即25*26*26.
model.add(Activation('relu'))
# 卷积参数是3*3=9
#池化
model.add(MaxPooling2D((2,2)))#以4个为一组，取最大的一个出来。获得25*13*13
# model.add(Dropout(0.25))
#第二次卷积
model.add(Conv2D(128,3,3,name='conv2_1'))#获得50*11*11
# 卷积参数是3*3*25=225
#第二次池化
model.add(MaxPooling2D(2,2))#获得50*5*5=1250
#flatten
model.add(Flatten())
#构建模型
model.add(Dense(units=100,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10,activation='softmax'))

#输出模型参数
model.summary()
#配置模型的学习过程
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#训练模型
result=model.fit(x_train,y_train,batch_size=100,epochs=10,validation_data=(x_test,y_test))
# model.save('E:/keras_data/mnist/first_CNN_model.h5')
model.save('E:/keras_data/mnist/first_CNN_model_2.h5')

#在trainning data 进行验证
# Trainscore=model.evaluate(x_train,y_train,batch_size=10000)
# print('\nTrain acc',Trainscore[1])

# score=model.evaluate(x_test,y_test,batch_size=10000)

#输出训练好的模型在测试集上的表现
# print('Total loss on Testing Set:',score[0])
# print('\nAccuracy of  Testing Set:',score[1])
#
# plt.figure()
# plt.plot(result.epoch,result.history['acc'],label="acc")
# plt.plot(result.epoch,result.history['val_acc'],label="val_acc")
# plt.scatter(result.epoch,result.history['acc'],marker='*')
# plt.scatter(result.epoch,result.history['val_acc'])
# plt.legend(loc='best')
# plt.show()
#
# plt.figure()
# plt.plot(result.epoch,result.history['loss'],label="loss")
# plt.plot(result.epoch,result.history['val_loss'],label="val_loss")
# plt.scatter(result.epoch,result.history['loss'],marker='*')
# plt.scatter(result.epoch,result.history['val_loss'],marker='*')
# plt.legend(loc='best')
# plt.show()