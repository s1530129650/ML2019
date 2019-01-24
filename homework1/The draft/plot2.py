#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: plot2.py
@time: 2019/1/24 10:58
"""
import numpy as np
import matplotlib.pyplot as plt

plt.figure()

plt.subplot(1,2,1)
plt.plot(range(10),range(10,20),'o-',ms =3,lw=1.5,color = 'black')
#plt.xlim(-200,-100)
#plt.ylim(-5,5)
plt.xlabel(r'$iter$',fontsize =16)
plt.ylabel(r'$loss$',fontsize =16)


#plt.subplot(1,2,2)
plt.figure()
plt.plot(range(10),range(10,20),'o-',ms =3,lw=1.5,color = 'green')
#plt.xlim(-200,-100)
#plt.ylim(-5,5)
plt.xlabel(r'$iter$',fontsize =16)
plt.ylabel(r'$loss$',fontsize =16)
plt.show()
