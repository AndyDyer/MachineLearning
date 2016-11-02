# -*- coding: utf-8 -*-
"""
andrew dyer
Titanic
"""
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import csv as csv

f = open('train.csv', 'rt')
data = []
try:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)
finally:
    f.close()
    
    ## DATA CLEANING
    
##Cabin Column deleted due to too many missing values
i = data[0].index("Cabin")
b = np.delete(data,i,1)
#Ticket row deleted due to irrelevent id
j = data[0].index("Ticket")
b = np.delete(b,j,1)

#Name row deleted due to irrelevent attribute
y = data[0].index("Name")
b = np.delete(b,y,1)



#Switching to Pandaa because ndarrays are lame.
pandaFrame = pd.DataFrame(data = b [1:,1:],
             index = b[1:,0,],
             columns = b[0,1:])
#drops tuples with missing data and replaces black data with NaN
pandaFrame = pandaFrame.replace('', np.nan, regex=True)
pandaFrame = pandaFrame.dropna() 
#converts fare to two sig figs
pandaFrame.Fare = pandaFrame.Fare.astype(float).round(2)
#converts age to whole int
pandaFrame.Age = pandaFrame.Age.astype(float).round(0).astype(int)

#binominalizes sex
pandaFrame = pandaFrame.replace({'Sex': {'female': '0'}}, regex=True)
pandaFrame = pandaFrame.replace({'Sex': {'male': '1'}}, regex=True)

#makes panda infer data types
pandaFrame = pandaFrame.convert_objects(convert_numeric=True)




def correlateMachineTrainer(pf, target_col):
    '''
     Args
    ----
    pf -- the pandaframe you want to search for training data.
    target_col -- the column you want to target.    
    -----
    '''
    corrPclass = target_col.corr(pf['Pclass'])
    corrSex = target_col.corr(pf['Sex'])
    corrAge = target_col.corr(pf['Age'])
    corrSibSp = target_col.corr(pf['SibSp'])
    corrParch = target_col.corr(pf['Parch'])
    corrFare = target_col.corr(pf['Fare'])
    
    data1 = [corrPclass, corrSex, corrAge, corrSibSp, corrParch, corrFare]
    cs = pd.DataFrame(data1, index=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'], dtype=float)
    cs['Abs'] = np.absolute(cs[0])
    
    #sorts correlation by their abs value greatest to weakest
    cs = cs.sort(['Abs'],ascending=False)
    #del cs['Abs']
    cs['Mean'] = [pf.Sex.mean(), pf.Pclass.mean(),pf.Fare.mean(), pf.Parch.mean(),
    pf.Age.mean(), pf.SibSp.mean()]
    
    return cs
    
    
    
correlateFrame = correlateMachineTrainer(pandaFrame, pandaFrame.Survived)


def bestGuess (pf, cf):
    corrSeries = cf[0]
    meanSeries = cf['Mean']
    print (corrSeries)
    print (meanSeries)
    pf['Survival % Guess'] = 0
    survguess = 0
    for i, row in pf.iterrows():
        myBestGuess = 0
        if row.Sex <= meanSeries[0]:
           myBestGuess = myBestGuess + 45
        if row.Pclass <= meanSeries[1]:
            myBestGuess = myBestGuess + 15
        if row.Fare >= meanSeries[2]:
            myBestGuess = myBestGuess + 5
            pf.loc[i, "Survival % Guess"] = myBestGuess
        if myBestGuess != 0:
            survguess = survguess + 1
    return pf    

pandaFrame = bestGuess(pandaFrame,correlateFrame)
print(pandaFrame)




