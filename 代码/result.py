# -*- coding: utf-8 -*-


import sklearn
import pandas as pd
import numpy as np
import VennABERS
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt


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



def plot():
    data=pd.read_csv("/home/shaoleshi/毕设/NSL_KDD-master/多模型协同/多模型roc.csv")
    #data2=pd.read_csv("/home/shaoleshi/民航/数据/kddcup.data/多模型协同/ivap_roc.csv")
    #data2=data2.sort_values(['fpr'], ascending = True)
    #print(data2.head())

    x=data["AutoEncoder_fpr"].tolist()
    y=data["AutoEncoder_tpr"].tolist()

    x1=data["HBOS_fpr"].tolist()
    y1=data["HBOS_tpr"].tolist()
    
    
    x2=data["IForest_fpr"].tolist()
    y2=data["IForest_tpr"].tolist()

    
    x3=data["KNN_fpr"].tolist()
    y3=data["KNN_tpr"].tolist()

    x4=data["LODA_fpr"].tolist()
    y4=data["LODA_tpr"].tolist()

    x5=data["LOF_fpr"].tolist()
    y5=data["LOF_tpr"].tolist()

    x6=data["MCD_fpr"].tolist()
    y6=data["MCD_tpr"].tolist()

    x7=data["OCSVM_fpr"].tolist()
    y7=data["OCSVM_tpr"].tolist()

    x8=data["PCA_fpr"].tolist()
    y8=data["PCA_tpr"].tolist()

    x9=data["CBLOF_fpr"].tolist()
    y9=data["CBLOF_tpr"].tolist()

    x10=data["VAE_fpr"].tolist()
    y10=data["VAE_tpr"].tolist()


    print(metrics.auc(x1, y1))
    print(metrics.auc(x2, y2))
    print(metrics.auc(x3, y3))
    print(metrics.auc(x4, y4))
    print(metrics.auc(x5, y5))
    print(metrics.auc(x6, y6))
    print(metrics.auc(x7, y7))
    print(metrics.auc(x8, y8))
    print(metrics.auc(x9, y9))
    print(metrics.auc(x10, y10))

    l=plt.plot(x, y, "pink", marker='*', ms=1,label="AutoEncoder")

    l1=plt.plot(x1, y1, "r", marker='*', ms=1,label="HBOS")
    l2=plt.plot(x2, y2, "y", marker='*', ms=1,label="IForest")
    l3=plt.plot(x3, y3, "c", marker='*', ms=1,label="KNN")
    l4=plt.plot(x4, y4, "m", marker='*', ms=1,label="LODA")
    l5=plt.plot(x5, y5, "g", marker='*', ms=1,label="LOF")
    l6=plt.plot(x6, y6, "b", marker='*', ms=1,label="MCD")
    l7=plt.plot(x7, y7, "k", marker='*', ms=1,label="OCSVM")
    l8=plt.plot(x8, y8, "greenyellow", marker='*', ms=1,label='PCA')
    l9=plt.plot(x9, y9, "sienna", marker='*', ms=1,label="CBLOF")
    l10=plt.plot(x10, y10, "orange", marker='*', ms=1,label="VAE")

    plt.legend()
    plt.title('NSL_KDD ROC')
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.show()
def iqr(result,i):
    
    Percentile=np.percentile(result["score"],[0,25,50,75,100])
    aa=result.score.tolist()
           
    IQR=0
    uplimit=0
    IQR=Percentile[3]-Percentile[1]
    uplimit=Percentile[3]+IQR*i
    
    if np.isnan(uplimit):
        aa=result.score.tolist()
        
        aa.sort(reverse = True)
        print(type(aa))
        IQR=aa[int(len(aa)/4)]-aa[int(len(aa)*3/4)]
        print(aa[int(len(aa)/4)])
        print(aa[int(len(aa)*3/4)])
        print(aa[1])
        uplimit=aa[int(len(aa)/4)]+IQR*i
    print(uplimit)
    print(len(result[result.score>uplimit]))
    

   
    return uplimit


def roc(test):
    f=[]
    t=[]
    dd = test.sort_values(by='score',ascending=False)
    print(dd.head())
    for i in range(1,112):
        
        #limit=iqr(train,0.01*i)  
        limit=list(dd.score)[i*100] 
        
        
        test['label_test']=test.score.apply(lambda x: 1 if x>limit else 0)
        fpr,tpr1,thresholds=sklearn.metrics.roc_curve(test.label_number,
                                  test.label_test,
                                  pos_label=None,
                                  sample_weight=None,
                                  drop_intermediate=True)
        
   
        f.append(list(fpr)[1])
        t.append(list(tpr1)[1])
    return f,t

def result(data,filename):
    dd = data.sort_values(by='score',ascending=False)
    #for i in range(6,10):
    train=pd.read_csv("/home/shaoleshi/毕设/NSL_KDD-master/多模型协同/AutoEncoder训练集.csv")
    limit=iqr(train,1.3) 
    
    data['label_test']=data.score.apply(lambda x: 1 if x>limit else 0)
    accuracy=accuracy_score(np.array(data.label_number),np.array(data.label_test))
    precision=precision_score(np.array(data.label_number),np.array(data.label_test))
    tpr=get_tpr(np.array(data.label_number),np.array(data.label_test))
    print("准确率为{}".format(accuracy))
    print("精确率为{}".format(precision))
    print("召回率为{}".format(tpr))
def read_csv(filepath,filename):
    data=pd.read_csv(filepath+filename)
    result(data,filename)
if __name__=='__main__':
    Model=["CBLOF","HBOS","PCA","MCD","IForest","LODA","LOF","KNN","OCSVM","VAE","AutoEncoder"]


    #多模型记录roc数据csv
    
    dict={}
    for i in Model:
        data=pd.read_csv("/home/shaoleshi/毕设/NSL_KDD-master/多模型协同/score/"+i+"测试集分数.csv")
        f,t=roc(data)
        dict[i+"_fpr"]=f
        dict[i+"_tpr"]=t
    data=pd.DataFrame(dict)
    data.to_csv("多模型roc.csv",index=None)
    
    '''
    #print("已完成.format{}",i)
    

    #计算准确率召回率
   
    filepath="/home/shaoleshi/民航/数据/kddcup.data/多模型协同/score/"
    filename="AutoEncoder测试集分数.csv"
    read_csv(filepath,filename)
    
    
　　　　'''
    
    plot()#总图
  
    
    
