# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:07:16 2016

@author: andrew
"""

from collections import Counter

Pos_list_words = []
Neg_list_words = []
stop_words = []

         
def getW(filepath):
    templist = []
    with open(filepath,'r') as f:
        for line in f:
            for word in line.split():
                templist.append(word)
    return templist
            
        
def StopWordRemover (StopList, MyList):
    for w in MyList:
        for q in StopList:
            if w == q:
                MyList.remove(w)
    return MyList

stop_words = getW('StopWords.txt')

Neg_list_words = getW('NegReviews.txt')

print ("neg list: ",len(Neg_list_words))
Neg_list_words = StopWordRemover(stop_words,Neg_list_words)
print ("list shrunk: ", len(Neg_list_words))

thefile = open('NegNoStop.txt', 'w')
for item in Neg_list_words:
  thefile.write("%s\n" % item)
thefile.close()

Pos_list_words = getW('PosReviews.txt')

print ("pos list: ",len(Pos_list_words))
Pos_list_words = StopWordRemover(stop_words,Pos_list_words)
print ("list shrunk: ", len(Pos_list_words))

thefile = open('PosNoStop.txt', 'w')
for item in Pos_list_words:
  thefile.write("%s\n" % item)
thefile.close()
