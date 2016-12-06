# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:30:24 2016
@author: andrew
"""
from collections import Counter
import pprint
import numpy as np
import pandas as pd
posList = []
negList = []
np.set_printoptions(precision=3, threshold=5)
def getW(filepath):
    templist = []
    with open(filepath,'r') as f:
        for line in f:
            templist.append(line)
    return templist
def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros
def freqScan(myCounter, myList, blank):
    for key in myCounter.keys():
        for index in range(len(myList)):
            if (myList[index] == key) :
                blank[index] = myCounter[key]
    return blank
def getBayes(myList,posDict,negDict,totalWords):
    posSum = 0
    negSum = 0
    for word in myList:
        if word in posDict:
            posSum += ((posDict[key] + 1)/(len(posDict)+totalWords))
        else:
            posSum += ((1)/(len(posDict)+totalWords))
        if word in negDict:
            negSum += ((negDict[key] + 1)/(len(negDict)+totalWords))
        else:
            negSum += ((1)/(len(negDict)+totalWords))

    return 1 if posSum > negSum else 0

posList = getW('PosNoStop.txt')
negList = getW('NegNoStop.txt')
totalList = posList + negList

totalUniqueWords = len(dict(Counter(totalList)))

totalCounter = dict(Counter(totalList))
for key in totalCounter.keys() :
    str = key.rstrip()
    totalCounter[str] = totalCounter.pop(key)
posCounter = dict(Counter(posList))
for key in posCounter.keys() :
    str = key.rstrip()
    posCounter[str] = posCounter.pop(key)
NegCounter = dict(Counter(negList))
for key in NegCounter.keys() :
    str = key.rstrip()
    NegCounter[str] = NegCounter.pop(key)
