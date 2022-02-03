# -*- coding: utf-8 -*-
import sklearn
import pandas as pd
import numpy as np
import VennABERS
from sklearn import metrics
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

def result(model,predict):
    data=pd.read_csv(predict)
    data=data[data[model]!=2]
    accuracy=accuracy_score(np.array(data['label_number']),np.array(data[model]))
    precision=precision_score(np.array(data['label_number']),np.array(data[model]))
    tpr=get_tpr(np.array(data['label_number']),np.array(data[model]))
    f1=2 * precision * tpr / (precision + tpr)
    print(model+"的结果如下所示:")
    print("准确率为{}".format(accuracy))
    print("精确率为{}".format(precision))
    print("召回率为{}".format(tpr))
    print("F1值为{}".format(f1))
    print("预测个数值为{}".format(len(data)))

if __name__=="__main__":

    Model=["PCA","MCD","IForest","LODA","LOF","KNN","OCSVM","CBLOF","HBOS","VAE"]
    predict="/home/shaoleshi/民航/数据/NSL_KDD-master/NSL_KDD-master/多模型协同/predict1.csv"

    result("VAE",predict)
    #for i in Model:
        #result(i,predict);
