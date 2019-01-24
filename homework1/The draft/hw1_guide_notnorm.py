#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: hw1_guide_notnorm.py
@time: 2019/1/24 19:32
"""
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
#import pandas as pd   #link https://docs.python.org/3/library/codecs.html#standard-encodings
"""
trainData = pd.read_csv('train.csv',encoding='big5')  # 读取训练数据 encoding='big5'c 繁体中文
print(trainData.shape)  # 返回csv文件形状
#print(type(trainData))

for item in trainData:
    print(item)
    break


print(trainData.iloc[1])
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import math

start = time.perf_counter()
#####################################
# 1. data loading
train = []
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
print("train data loading:",train[0][:12])

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

print("the shape of test data ",xTest.shape)


#####################################
'''
#3.归一化normalize
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

print("train data after normalize:",train[1,:12])


for i in range(xTest.shape[1]):
    xTest[:, i] = MaxMinNormalization(xTest[:, i], Max[i//9], Min[i//9])

print("test data after normalize:",xTest[:12,1])
'''
#####################################
# 2.pre-processing

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

#print("after append,xTrain [0]:",xTrain[0][:10])

x = np.array(xTrain)
y = np.array(yTrain)

x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

#np.savetxt("x_value.txt", x)


#####################################
#4.training

l_rate = 10
repeat = 10000

w = l_rate*np.ones(len(x[0])) #162*1
x_t = x.transpose()  #x: 5652*162  ; x_t:162*5652
loss_history = []
s_gra = np.zeros(len(x[0]))


x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

for i in range(repeat):
    hypo = np.dot(x,w)
    loss = hypo - y
    cost = np.sum(loss**2) / len(x)
    loss_history.append(cost)
    cost_a  = math.sqrt(cost)
    gra = np.dot(x_t,loss)

    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra/ada

print("loss history is",loss_history)
np.save('model.npy',w)

#np.savetxt("w_value.txt",w)
#w = np.load('model.npy')


############################################
#5 testing
test_x = np.concatenate((np.ones((xTest.shape[0],1)),xTest), axis=1)

ans = []
predict =[]
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)
predict = np.dot(test_x,w)
#predict = np.dot(test_x,w)

##########################
#6 evaluate
truefile = pd.read_csv("ans.csv")             #读取csv文件
true = list(truefile ["value"])


predictError = np.sum(abs(predict - true))/(len(predict))
print("predict error is :",predictError )
end = time.perf_counter()
print ('time cost is ',end-start)


plt.plot(range(repeat),loss_history,'o-',ms =3,lw=1.5,color = 'black')
#plt.xlim(-200,-100)
plt.ylim(0,200)
plt.xlabel(r'$iter$',fontsize =12)
plt.ylabel(r'$loss$',fontsize =12)
plt.show()

##################

filename = "predictnotnorm.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()

