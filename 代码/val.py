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



def handle(test,j):
    dd = test.sort_values(by='score',ascending=False)
    print(dd.head())
    max=0
    flag=0
    for i in range(1,112):  
        limit=list(dd.score)[i*100]
        test['label_test']=test.score.apply(lambda x: 1 if x>limit else 0)
        precision=precision_score(np.array(test.label_number),np.array(test.label_test))
        tpr=get_tpr(np.array(test.label_number),np.array(test.label_test))
        f1=2 * precision * tpr / (precision + tpr)
        if max<f1:
            max=f1
            flag=limit
    data=pd.read_csv("/home/shaoleshi/毕设/NSL_KDD-master/多模型协同/score/"+j+"测试集分数.csv")
    data['label_test']=data.score.apply(lambda x: 1 if x>flag else 0)
    accuracy=accuracy_score(np.array(data.label_number),np.array(data.label_test))
    precision=precision_score(np.array(data.label_number),np.array(data.label_test))
    tpr=get_tpr(np.array(data.label_number),np.array(data.label_test))
    f1=2 * precision * tpr / (precision + tpr)
    print(j+"的结果如下所示:")
    print("准确率为{}".format(accuracy))
    print("精确率为{}".format(precision))
    print("召回率为{}".format(tpr))
    print("F1值为{}".format(f1))




if __name__=='__main__':
    Model=["PCA","MCD","IForest","LODA","LOF","KNN","CBLOF","HBOS","VAE","OCSVM","AutoEncoder"]
    #Model=["CBLOF"]
    for i in Model:
        data=pd.read_csv("/home/shaoleshi/毕设/NSL_KDD-master/多模型协同/score/"+i+"验证集分数.csv")
        handle(data,i)
   

