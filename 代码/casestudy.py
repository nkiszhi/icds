# -*- coding: utf-8 -*-
import sklearn
import pandas as pd
import numpy as np
import VennABERS
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
import os

#找例子
'''
data=pd.read_csv("./predict1.csv")
#data.columns=["label","PCA","MCD","IForest","LODA","LOF","KNN","CBLOF","HBOS","VAE","OCSVM"]
print(data.head())
for indexs in data.index:
    data1=data.loc[indexs].values[0:-1]
    data1=data.loc[indexs].tolist()
    count0=data1[0]
    count1=data1[1:].count(1)
    count2=data1[1:].count(2)
    count3=data1[1:].count(0)
    if count1==3 and count0==1 and count3==1:
    #if count3==count1 and count2>0:
        print(indexs) 
    
'''
def plot(f,t):
    plt.plot(f, t, "r", marker='*', ms=1, label="a")
    plt.xlabel("p1-p0")
    plt.ylabel("f1")
    plt.show()

global t

def accuracy_score(y, y_hat):
    return sum(yi == yi_hat for yi, yi_hat in zip(y, y_hat)) / len(y)

def precision_score(y, y_hat):
   
    true_positive = sum(yi and yi_hat for yi, yi_hat in zip(y, y_hat))
    predicted_positive = sum(y_hat)
    return true_positive / predicted_positive

def get_tpr(y, y_hat):
    true_positive = sum(yi and yi_hat for yi, yi_hat in zip(y, y_hat))
    actual_positive = sum(y)

    return true_positive / actual_positive
def count_p(p0,p1):
    '''
    global t
    global yz
    if p1-p0>=yz-0.000000001:
        return 2
    '''
    #if p1-p0>=t:
        #return 2

    if p1/(1-p0+p1)>(0.5-t) and p1/(1-p0+p1)<(0.50+0.5*t):
        return 2
    
    if p1/(1-p0+p1)>0.5: #可以用来调节roc
        return 1
    else:
        return 0


data=pd.read_csv("./p01.csv")
xx=[]
yy=[]

for i in range(1,50):
    t=0.005*i

    data['venn_pre']=data.apply(lambda x:count_p(x["PCA_p0"],x["PCA_p1"]),axis=1)
    print(len(data))
    data1=data[data['venn_pre']!=2]
    print(len(data1))
    try:
        accuracy=accuracy_score(np.array(data1['label_number']),np.array(data1['venn_pre']))
        precision=precision_score(np.array(data1['label_number']),np.array(data1['venn_pre']))
        tpr=get_tpr(np.array(data1['label_number']),np.array(data1['venn_pre']))
        f1=2 * precision * tpr / (precision + tpr)

    except ZeroDivisionError:
        continue
    else:
        xx.append(t)
        yy.append(f1)       
        
    '''

    print("准确率为{}".format(accuracy))
    print("精确率为{}".format(precision))
    print("召回率为{}".format(tpr))
    '''
    print("F1值为{}".format(f1))
    

plot(xx,yy)






