#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: wordCount.py
@time: 2019/1/16 16:48
"""
import json
from collections import Counter
with open('hw0_data\words.txt', 'r') as f:
    wordList = f.read().split()
print(wordList)
wordCount = Counter(wordList)
wordDict =  dict(wordCount)
length = len(wordDict)
i = 0
#js = json.dumps(wordDict)
with open('Q1.txt','w') as file:
    #file.write(js)
    for word,fre in wordDict.items():
        i += 1
        #print(word)
        if i == length:
            file.write((word) + " " + str(fre) )
        else :
            file.write((word)+" "+str(fre)+'\n')
