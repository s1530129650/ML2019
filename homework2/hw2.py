#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: hw2.py
@time: 2019/2/18 14:27
"""

# import library
import pandas as pd
import numpy as np
from math import log, floor

# 1. data loading

X_train = pd.read_csv("./Data/X_train",sep =',',header=0) #sep参数是用于指定分割符
X_test = pd.read_csv("./Data/X_test",sep = ',', header = 0)
Y_train = pd.read_csv("./Data/Y_train",sep =',',header=0)
#print(X_train.info())
#print(X_train.columns )
X_train = X_train.values
X_test = X_test.values
Y_train = Y_train.values
#print("Y_train shape is ",Y_train.shape)
#print("X_train is ",X_train.shape)
#print("Y_train is ",Y_train)
'''
df=X_train.values
print("type of df ",type(df))
print('df:',df)
X_train = np.array(X_train)
print("type of x_train ",type(X_train))
print("x_train is ",X_train)
'''
# 2.feature extraction



# 3.normalize

X_train_test = np.concatenate((X_train,X_test))
mean = X_train_test.mean(axis = 0)
sigma = X_train_test.std(axis = 0)
mean = np.tile(mean,(X_train_test.shape[0],1))
sigma = np.tile(sigma,(X_train_test.shape[0],1))
X_train_test_normed = (X_train_test - mean)/sigma
X_train = X_train_test_normed[0:X_train.shape[0]]
X_test = X_train_test_normed[X_train.shape[0]:]
#print(" X_train after normalize is :", X_train)
#print(mean.shape) #106

# 4.shuffle
def _shuffle(X,Y):
    randomsize = np.arange(len(X))
    # print(randomsize)
    np.random.shuffle(randomsize)
    return (X[randomsize],Y[randomsize])


randomsize = np.arange(X_train.shape[0])
#print(randomsize)
np.random.shuffle(randomsize)
X_train , Y_train = X_train[randomsize],Y_train[randomsize]
#print("after shufleing is ",randomsize)
#print(type(randomsize))


# 5. split validation set

percent = 0.1
all_data_size = len(X_train)
valid_data_size = int(floor(all_data_size*percent))
X_train_valid , Y_train_valid = X_train[0:valid_data_size] , Y_train[0:valid_data_size]
X_train_train , Y_train_train = X_train[valid_data_size:] , Y_train[valid_data_size:]

#define sigmoid

def sigmoid(z):
    res = 1 / (1 + np.exp(-z))
    #numpy.clip(a, a_min, a_max, out=None)[
    return np.clip(res,1e-8,1-(1e-8))
#get valid score
def valid(w,b,X_valid,Y_valid):
    valid_data_size = len(X_valid)
    z = (np.dot(X_valid,np.transpose(w)) + b )
    y =  sigmoid (z)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    acc = float(result.sum())/valid_data_size
    print("Validation acc = %f"%(acc))
    return acc
# 6. train

#Initiallize parameter, hyperparameter
w = np.zeros((106,))
b = np.zeros((1,))
l_rate = 0.005
batch_size = 64
train_data_size = len(X_train_train)
step_num  = int(floor(train_data_size / batch_size))
#step_num  = 1
epoch_num = 5000
#epoch_num = 2
save_param_iter = 50
valid_acc = []
#Start training
total_loss = 0.0
for epoch in range(1,epoch_num):
    if(epoch) % save_param_iter == 0:
        print('====Saving Param at epoch %d====' % epoch)

        print('epoch avg loss = %f'% (total_loss/(float(save_param_iter)* train_data_size)))

        total_loss = 0.0
        acc = valid(w,b,X_train_valid,Y_train_valid)
        valid_acc.append(acc)

    #Random shuffle
    X_train_train,Y_train_train = _shuffle(X_train_train,Y_train_train)

    #Train with batch
    for idx in range(step_num):
        X = X_train_train[idx*batch_size:(idx+1)*batch_size]
        Y = Y_train_train[idx*batch_size:(idx+1)*batch_size]

        z = np.dot(X, np.transpose(w)) + b
        y = sigmoid(z)
        #print('shape of Y',Y.shape)
        #print("sahpe of y",y.shape)
        cross_entropy = -1*(np.dot(np.squeeze(Y),np.log(y))+np.dot((1-np.squeeze(Y)),(np.log(1-y))) )
        total_loss += cross_entropy
        #print("np.squeeze(Y) - y",np.squeeze(Y) - y)
        #print("(np.squeeze(Y) - y).reshape((batch_size,1)",(np.squeeze(Y) - y).reshape((batch_size,1)))

        w_grad = np.mean(-1*X*(np.squeeze(Y) - y).reshape((batch_size,1)),axis = 0)
        b_grad = np.mean(-1 * (np.squeeze(Y) - y ))
        #print("w_grad",w_grad)
        #print(w_grad.shape)

        #SGD Updating parameters
        w = w - l_rate * w_grad
        b = b - l_rate * b_grad

