# -*- coding: utf-8 -*-
import sklearn
import pandas as pd
import numpy as np
import VennABERS
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
import os

global yz

def plot(f,t):
    plt.plot(f, t, "r", marker='*', ms=1, label="a")
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.show()

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
    #if p1-p0>=0.3:
        #return 2

    if p1/(1-p0+p1)>0.25 and p1/(1-p0+p1)<0.60:
        return 2
    
    if p1/(1-p0+p1)>0.5: #可以用来调节roc
        return 1
    else:
        return 0


def count_re(a,b,c,d,e,f,g,h,i,j):
    T=(a,b,c,d,e,f,g,h,i,j)
    r0=T.count(0)
    r1=T.count(1)
    if r0==0 and r1==0:
        return 2
    if r1>r0:
        return 1
    else:
        return 0


def p_yuzhi(p0,p1):
    return p1-p0

def ivap(p01,model):
    global yz
    '''
    p0p1="/home/shaoleshi/民航/数据/NSL_KDD-master/NSL_KDD-master/多模型协同/阈值.csv"
    data2=pd.read_csv(p0p1)
    yz=data2[model+"_p"].tolist()[0]
    print(yz)
    '''

    data=pd.read_csv(p01)
    data[model]=data.apply(lambda x:count_p(x[model+"_p0"],x[model+"_p1"]),axis=1)
    pre="/home/shaoleshi/民航/数据/NSL_KDD-master/NSL_KDD-master/多模型协同/predict1.csv"
    if os.path.exists(pre):
        p_data=pd.read_csv(pre)
        p_data[model]=data[model]
        p_data.to_csv('predict1.csv',index=None)
    else:
        p_data=pd.merge(data['label_number'],data[model],left_index=True,right_index=True)
        p_data.to_csv('predict1.csv',index=None)

def result(predict):
    data=pd.read_csv(predict)
    data['venn_pre']=data.apply(lambda x:count_re(x["PCA"],x["MCD"],x["IForest"],x["LODA"],x["LOF"],x["KNN"],x["OCSVM"],x["CBLOF"],x["HBOS"],x["VAE"]),axis=1)

    print(len(data))
    data=data[data['venn_pre']!=2]
    print(len(data))
    accuracy=accuracy_score(np.array(data['label_number']),np.array(data['venn_pre']))
    precision=precision_score(np.array(data['label_number']),np.array(data['venn_pre']))
    tpr=get_tpr(np.array(data['label_number']),np.array(data['venn_pre']))
    f1=2 * precision * tpr / (precision + tpr)

    print("ivap的结果如下所示:")
    print("准确率为{}".format(accuracy))
    print("精确率为{}".format(precision))
    print("召回率为{}".format(tpr))
    print("F1值为{}".format(f1))
    #return list(fpr)[1],list(tpr1)[1]


#处理p1-p0阈值的问题
def p1p0(p01,model):
    i=0
    data=pd.read_csv(p01)
    data[model+'_p']=data.apply(lambda x:p_yuzhi(x[model+"_p0"],x[model+"_p1"]),axis=1)
    #pre="/home/shaoleshi/民航/数据/kddcup.data/多模型协同/阈值.csv"
    list1=data[model+'_p'].tolist()
    list1=sorted(list1)
    if model=="PCA":
        i=0.89
    if model=="MCD":
        i=0.95
    if model=="IForest":
        i=0.99
    if model=="LODA":
        i=0.93
    if model=="LOF":
        i=0.7
    if model=="KNN":
        i=0.7
    if model=="OCSVM":
        i=0.7
    if model=="CBLOF":
        i=0.6
    if model=="HBOS":
        i=0.6
    if model=="VAE":
        i=0.6
    
    t=list1[int(len(list1)*i)]
    '''
    print(list1[int(len(list1)*0.1)])
    print(list1[int(len(list1)*0.25)])
    print(list1[int(len(list1)*0.5)])
    print(list1[int(len(list1)*0.75)])
    print(list1[int(len(list1)*0.99)])
    '''
    return t



#多模型投票所得结果
def result1(predict,model):
    data=pd.read_csv(predict)
    #print(len(data))
    #data=data[data[model]!=2]
    #print(len(data))
    print("模型{}结果为".format(model))
    accuracy=accuracy_score(np.array(data['label_number']),np.array(data[model]))
    precision=precision_score(np.array(data['label_number']),np.array(data[model]))
    tpr=get_tpr(np.array(data['label_number']),np.array(data[model]))
    f1=2 * precision * tpr / (precision + tpr)
    print("准确率为{}".format(accuracy))
    print("精确率为{}".format(precision))
    print("召回率为{}".format(tpr))
    print("F1值为{}".format(f1))


if __name__=="__main__":
    global t
    f=[]
    tt=[]

    Model=["PCA","MCD","IForest","LODA","LOF","KNN","OCSVM","CBLOF","HBOS","VAE"]
    p01="/home/shaoleshi/民航/数据/NSL_KDD-master/NSL_KDD-master/多模型协同/p01.csv"
    predict="/home/shaoleshi/民航/数据/NSL_KDD-master/NSL_KDD-master/多模型协同/predict.csv"
    filepath='/home/shaoleshi/民航/数据/NSL_KDD-master/NSL_KDD-master/多模型协同/score/'
    predict1="/home/shaoleshi/民航/数据/NSL_KDD-master/NSL_KDD-master/多模型协同/predict1.csv"

    '''
    #设置p1-p0参数
    dict_p={}
    for i in Model:
        t=p1p0(p01,i)
        dict_p[i+'_p']=t
    ppp=pd.DataFrame(dict_p,index=[0])
    print(dict_p)
    ppp.to_csv("阈值.csv",index=None)
    
    #ivap(p01,"IForest") #0.85
    #result1(predict1,"IForest")

    #ivap(p01,"PCA") #0.6
    #result1(predict1,"PCA")    

    #ivap(p01,"LOF") #0.7
    #result1(predict1,"LOF")
  
    '''
    #输出每个模型的预测结果
    for i in Model:
        ivap(p01,i)
        #result1(predict1,i)


    result(predict1)
   

    

    '''
    #设置p1-p0参数
    dict_p={}
    for i in Model:
        t=p1p0(p01,i)
        dict_p[i+'_p']=t
    ppp=pd.DataFrame(dict_p,index=[0])
    print(dict_p)
    ppp.to_csv("阈值.csv",index=None)
    '''
   

