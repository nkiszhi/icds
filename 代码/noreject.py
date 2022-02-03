# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import VennABERS
import os
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

def count_re(a,b,c,d,e,f,g,h,i,j,k):
    T=(a,b,c,d,e,f,g,h,i,j,k)
    r0=T.count(0)
    r1=T.count(1)
    if r1>r0:
        return 1
    else:
        return 0

def count_p(p0,p1):
    #if p1-p0>=0.02:
        #return 2
    if p1/(1-p0+p1)>0.18: #可以用来调节roc
        return 1
    else:
        return 0


def read_csv(filepath,testname,calname):
    name=calname.split('校')[0]
    test_data=pd.read_csv(filepath+testname)
    cal_data=pd.read_csv(filepath+calname)
    cal_data['new_col']= list(zip(cal_data.score,cal_data.label_number))
    tac=cal_data.new_col.tolist()
    tec=test_data.score.tolist()
    p0,p1=VennABERS.ScoresToMultiProbs(tac,tec)
    test_data[name+'_p0']=p0.tolist()
    test_data[name+'_p1']=p1.tolist()
    #p_data=pd.merge(test_data[name+'_p0'],test_data[name+'_p1'],left_index=True,right_index=True)
    
    p01="/home/shaoleshi/毕设/NSL_KDD-master/多模型协同/p01.csv"
    if os.path.exists(p01):
        p_data=pd.read_csv(p01)
        p_data[name+'_p0']=test_data[name+'_p0']
        p_data[name+'_p1']=test_data[name+'_p1']
        p_data.to_csv('p01.csv',index=None)
    else: 
        p_data=pd.merge(test_data[name+'_p0'],test_data[name+'_p1'],left_index=True,right_index=True)
        p_data.to_csv('p01.csv',index=None)

def ivap(p01,model):
    data=pd.read_csv(p01)
    data[model]=data.apply(lambda x:count_p(x[model+"_p0"],x[model+"_p1"]),axis=1)
    pre="/home/shaoleshi/毕设/NSL_KDD-master/多模型协同/predict.csv"
    if os.path.exists(pre):
        p_data=pd.read_csv(pre)
        p_data[model]=data[model]
        p_data.to_csv('predict.csv',index=None)
    else:
        p_data=pd.merge(data['label_number'],data[model],left_index=True,right_index=True)
        p_data.to_csv('predict.csv',index=None)

def result(predict):
    data=pd.read_csv(predict)
    #data2=pd.read_csv("/home/shaoleshi/毕设/NSL_KDD-master/多模型协同/p01.csv")
    data['venn_pre']=data.apply(lambda x:count_re(x["CBLOF"],x["HBOS"],x["VAE"],x["PCA"],x["MCD"],x["IForest"],x["LODA"],x["AutoEncoder"],x["LOF"],x["KNN"],x["OCSVM"]),axis=1)
   
    accuracy=accuracy_score(np.array(data['label_number']),np.array(data['venn_pre']))
    precision=precision_score(np.array(data['label_number']),np.array(data['venn_pre']))
    tpr=get_tpr(np.array(data['label_number']),np.array(data['venn_pre']))
    f1=2 * precision * tpr / (precision + tpr)
    print(data.head())
    print(len(data['label_number']))
    print("准确率为{}".format(accuracy))
    print("精确率为{}".format(precision))
    print("召回率为{}".format(tpr))
    print("F1值为{}".format(f1))



if __name__=="__main__":
    Model=["CBLOF","HBOS","PCA","MCD","IForest","LODA","LOF","KNN","OCSVM","VAE","AutoEncoder"]
    p01="/home/shaoleshi/毕设/NSL_KDD-master/多模型协同/p01.csv"
    predict="/home/shaoleshi/毕设/NSL_KDD-master/多模型协同/predict.csv"
    filepath='/home/shaoleshi/毕设/NSL_KDD-master/多模型协同/score/'
    '''
    for i in Model:
        testname=i+'测试集分数.csv'
        calname=i+'校准集分数.csv'
        read_csv(filepath,testname,calname)
    '''
    for i in Model:
        ivap(p01,i)
        print(i)
   
    result(predict)
    
    
