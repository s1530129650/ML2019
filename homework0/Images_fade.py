#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: Images_fade.py
@time: 2019/1/16 17:17
"""
from PIL import Image
import numpy as np

filename = "hw0_data/westbrook.jpg"
im=Image.open(filename) #open the  image

imgs = np.array(im) #transform to array


imgsDiv2 = np.trunc(imgs/2)
imgInt = imgsDiv2.astype(np.int)
imgInt = imgInt[:,:,:3]

finalImg = Image.fromarray(np.uint8(imgInt))
finalImg.save("Q2.jpg")
#注意img如果是uint16的矩阵而不转为uint8的话，Image.fromarray这句会报错

