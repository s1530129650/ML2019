#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""

@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence
@contact: 123@qq.com
@site:
@software: PyCharm
@file: hw1.py
@time: 2019/1/23 19:24
"""

# 0 import library
import csv  #read csv file
import numpy as np  # process array
import matplotlib.pyplot as plt #plot
import pandas as pd   # read csv file
import time # calculate running time
import math #  Take the square root

start = time.perf_counter()
#####################################
# 1. data loading
train = []  #对于train，每一行存储一种指标的数据，总共18行，由于数据有5760=24*20*12个，所以train为18行，5760列
test = []
for i in range(18):
    train.append([])

with open('train.csv','r',encoding='big5') as trainFile:
    trainData = csv.reader(trainFile)
    count = 0
    for row in trainData:
        if count == 0:
            count += 1
            continue
        for i in range(3,27):
            if row[i] == "NR":
                train[(count-1)%18].append(float(0))
            else:
                train[(count - 1) % 18].append(float(row[i]))
        count += 1

#对于test，直接提取feature，也就是把9个小时的18个指标，共计162个特征。
#一个数据point为一行，240个数据点就是240行，每行162个特征
with open('test.csv','r',encoding='big5') as testFile:
    testData = csv.reader(testFile)
    count = 0
    for row in testData:
        if count % 18 == 0:
            test.append([])
            for i in range(2,11):
                test[(count) // 18].append(float(row[i]))
        else:
            for i in range(2, 11):
                if row[i] == "NR":
                    test[(count)//18].append(float(0))
                else:
                    test[(count ) // 18].append(float(row[i]))
        count += 1
xTest = np.array(test)




#####################################
'''
3.归一化normalize
归一化使用的公式很简单，就是x1 = （x- min）/(max - min),这里为了统一我们会保存每个指标的最大最小值，
便于对test做normalization。
其实不用归一化，效果也很好.
'''
# normalize training set
train = np.array(train)
def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min)
    return x
Max = []
Min = []
for i in range(len(train)):
    Max.append(train[i,:].max())
    Min.append(train[i,:].min())
    train[i, :] = MaxMinNormalization(train[i,:],Max[i],Min[i])
# normalize testing set
for i in range(xTest.shape[1]):
    xTest[:, i] = MaxMinNormalization(xTest[:, i], Max[i//9], Min[i//9])

#####################################
# 2.pre-processing
#主要是把training data ，parse to (x,y)
#即一个数据point为一行，5760个数据点就是5760行，每行162个特征
xTrain = []
yTrain = []

lenOfMonth = int(len(train[0]) / 12)  #5760/12 =480 = 20*24
lenOfMonthX = lenOfMonth - 9
#per month
# i-> month; j ->hours in a month; k-> feature; t -> 9 hours
for i in range(12):
    for j in range(lenOfMonthX):
        xTrain.append([])
        for k in range(18):
            for t in range(9):
                xTrain[lenOfMonthX * i +j].append(train[k][lenOfMonth * i +j + t])
                # x是把9个小时的指标按照[九个小时的指标1，九个小时的指标2，]组成了一个vetors
        yTrain.append(train[9][lenOfMonth*i+j+9])

x = np.array(xTrain)
y = np.array(yTrain)
#加上一个bias b
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

#np.savetxt("x_value.txt", x)  #保存数据值，方便查看


#####################################
#4.training
# 这个过程主要是梯度下降的法

l_rate = 10  #使用了AdaGrad，所以learning rate不用过分的调整，不然是要做调整的。
repeat = 10000

w = l_rate*np.ones(len(x[0])) #这样子初始化防止一开始出现尖峰，具体描述见博客https://mp.csdn.net/mdeditor/86614438
x_t = x.transpose()  #x: 5652*162  ; x_t:162*5652 转置
loss_history = []
s_gra = np.zeros(len(x[0]))

x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

for i in range(repeat):
    hypo = np.dot(x,w)
    loss = hypo - y
    cost = np.sum(loss**2) / len(x)
    loss_history.append(cost)
    gra = np.dot(x_t,loss)

    s_gra += gra**2 #将历史的梯度平方相加
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra/ada #AdaGrad

print("loss history is",loss_history)
np.save('model_norm_AdaGrad.npy',w)

#save/read model
#np.savetxt("w_value.txt",w)
#w = np.load('model.npy')

############################################
#5 testing
#预测就是 y = w^T*x + b ,如果做了归一化，需要还原y
test_x = np.concatenate((np.ones((xTest.shape[0],1)),xTest), axis=1)
#为了满足输出要求，可以在for循坏中一步步计算，也可以像predict一样直接计算
ans = []
predict =[]
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])*(Max[9] - Min[9]) + Min[9]
    ans[i].append(a)

predict = np.dot(test_x,w)*(Max[9] - Min[9]) + Min[9]
#predict = np.dot(test_x,w)

##########################
#6 evaluate
truefile = pd.read_csv("ans.csv")             #读取csv文件
true = list(truefile["value"])       #pandas处理csv的好处


predictError = np.sum(abs(predict - true))/(len(predict))
print("predict error is :",predictError )
end = time.perf_counter()
print ('time cost is ',end-start)

#plot
plt.plot(range(repeat),loss_history,'o-',ms =3,lw=1.5,color = 'black')
#plt.xlim(-200,-100)
plt.ylim(0,0.02)
plt.xlabel(r'$iter$',fontsize =12)
plt.ylabel(r'$loss$',fontsize =12)
plt.show()

##################

filename = "predict_norm_AdaGrad.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()
