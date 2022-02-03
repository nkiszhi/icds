# -*- coding: utf-8 -*-
import sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
'''
author:leshi

'''
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.lscp import LSCP
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.pca import PCA
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.cof import COF
from pyod.models.mcd import MCD
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.cblof import CBLOF
from pyod.models.loda import LODA
from pyod.models.ocsvm import OCSVM
from pyod.models.abod import ABOD
from pyod.models.vae import VAE
from pyod.models.so_gaal import SO_GAAL
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.loci import LOCI

COF_clf = COF(contamination=0.01,n_neighbors=30) 
LSCP_clf = LSCP(contamination=0.01,detector_list = [LOF(), PCA()])
LOCI_clf = LOCI(contamination=0.05)




VAE_clf = VAE(contamination=0.001, epochs=50, gamma=0.8, capacity=0.2, encoder_neurons=[9, 4], decoder_neurons=[4, 9])
ABOD_clf = ABOD(contamination=0.01,n_neighbors=20,method='default')
FeatureBagging_clf = FeatureBagging(contamination=0.01,)
AutoEncoder_clf = AutoEncoder(contamination=0.001)
OCSVM_clf= OCSVM(contamination=0.001)
LODA_clf = LODA(contamination=0.001)
CBLOF_clf = CBLOF(contamination=0.001)
LOF_clf = LOF(contamination=0.001)
PCA_clf = PCA(contamination=0.001)
HBOS_clf = HBOS(contamination=0.001)
IForest_clf = IForest(contamination=0.001)
MCD_clf = MCD(contamination=0.001)
KNN_clf = KNN(contamination=0.001)
SO_GAAL_clf = SO_GAAL(contamination=0.001)

MO_GAAL_clf = MO_GAAL(contamination=0.05, stop_epochs=2) #需要调参

Path="/home/shaoleshi/毕设/NSL_KDD-master/多模型协同/"
Model_list=["CBLOF","HBOS","PCA","MCD","IForest","LODA","LOF","KNN","OCSVM","VAE","AutoEncoder"]
#Model_list=["VAE"]
def read_csv(filepath,filename):
    total_data=pd.read_csv(filepath+filename,header=None)
    total_data.columns=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
                        "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell",
                         "su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login"
                        ,"is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate"
                        ,"same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate"
                         ,"dst_host_same_src_port_rate:","dst_host_srv_diff_host_rate","dst_host_serror_rate"
                         ,"dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","biao"]
    train1,cal=np.split(total_data.sample(frac=1),[int(.4*len(total_data))])
    filename2="KDDTest+.csv"
    test_data=pd.read_csv(filepath+filename2,header=None)
    #cal=cal.sample(n=10000,random_state=2)
    train=train1[train1["label"]=="normal"].sample(n=20000,random_state=1)
    cal1=cal[cal["label"]=="normal"].sample(n=15000,random_state=1)
    cal2=cal[cal["label"]!="normal"].sample(n=15000,random_state=10)
    cal = cal1.append(cal2)
    test_data.columns=total_data.columns
    val,test1=np.split(test_data.sample(frac=1),[int(.5*len(test_data))])
    val.to_csv(Path+"验证集.csv",index=None)
    test1.to_csv(Path+"测试集.csv",index=None)
    train.to_csv(Path+"训练集.csv",index=None)
    cal.to_csv(Path+"校准集.csv",index=None)#训练模型得到模型

def count_data():
    Path="/home/shaoleshi/毕设/NSL_KDD-master/多模型协同/"
    sample_ab=pd.read_csv(Path+"测试集.csv")
    sample_nor=pd.read_csv(Path+"训练集.csv")
    sample_cal=pd.read_csv(Path+"校准集.csv")
    sample_val=pd.read_csv(Path+"验证集.csv")
    
    nor=0
    abn=0
    nor_va=len(sample_val[sample_ab["label"]=="normal"])
    ab_va=len(sample_val[sample_ab["label"]!="normal"])
    print("验证")
    print(nor_va)
    print(ab_va)
    nor=nor+len(sample_ab[sample_ab["label"]=="normal"])
    abn=abn+len(sample_ab[sample_ab["label"]!="normal"])

    nor=nor+len(sample_nor[sample_nor["label"]=="normal"])
    abn=abn+len(sample_nor[sample_nor["label"]!="normal"])
    nor=nor+len(sample_cal[sample_cal["label"]=="normal"])
    abn=abn+len(sample_cal[sample_cal["label"]!="normal"])

    print(nor)
    print(abn)

#训练模型得到校准集测试集文件
def model_train(model):
    total_data=pd.read_csv(Path+"训练集.csv")
    x_train=total_data.drop(columns=["protocol_type","service","flag","src_bytes","label","biao"],axis=1)#训练去除无用列名
    x_train = pd.DataFrame(normalize(x_train.values), index=x_train.index, columns=x_train.columns)
    clf=eval(model+"_clf").fit(x_train)
    y_train_scores = clf.decision_scores_
    total_data['score']=y_train_scores
    total_data.to_csv(model+"训练集.csv",index=None)
    model_cal(clf,model)
    model_test(clf,model)
    model_val(clf,model)
def model_cal(clf,model):
    total_data=pd.read_csv(Path+"校准集.csv")
    x_cal=total_data.drop(columns=["protocol_type","service","flag","src_bytes","label","biao"],axis=1)
    x_cal = pd.DataFrame(normalize(x_cal.values), index=x_cal.index, columns=x_cal.columns)
    y_cal_scores = clf.decision_function(x_cal)
    total_data['label_number']=total_data.label.apply(lambda x: 0 if x=="normal" else 1)
    total_data['score']=y_cal_scores
    total_data.to_csv(Path+"score/"+model+"校准集分数.csv",index=None)#记录校准集分数
def model_test(clf,model):
    total_data=pd.read_csv(Path+"测试集.csv")
    x_test=total_data.drop(columns=["protocol_type","service","flag","src_bytes","label","biao"],axis=1)
    x_test = pd.DataFrame(normalize(x_test.values), index=x_test.index, columns=x_test.columns)
    y_test_scores = clf.decision_function(x_test)
    total_data['label_number']=total_data.label.apply(lambda x: 0 if x=="normal" else 1)
    total_data['score']=y_test_scores
    total_data.to_csv(Path+"score/"+model+"测试集分数.csv",index=None)#记录校准集分数

def model_val(clf,model):
    total_data=pd.read_csv(Path+"验证集.csv")
    x_test=total_data.drop(columns=["protocol_type","service","flag","src_bytes","label","biao"],axis=1)
    x_test = pd.DataFrame(normalize(x_test.values), index=x_test.index, columns=x_test.columns)
    y_test_scores = clf.decision_function(x_test)
    total_data['label_number']=total_data.label.apply(lambda x: 0 if x=="normal" else 1)
    total_data['score']=y_test_scores
    total_data.to_csv(Path+"score/"+model+"验证集分数.csv",index=None)#记录校准集分数
if __name__=="__main__":

    '''
    filepath='/home/shaoleshi/毕设/NSL_KDD-master/'
    filename='KDDTrain+.csv'
    read_csv(filepath,filename)#提取训练集，测试集，校准集
    '''
    #count_data()
    
    for i in Model_list:
        model_train(i)
        print("已完成.format{}",i)
    
    
    
