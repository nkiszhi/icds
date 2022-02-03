# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import VennABERS

def read_csv(filepath,testname,trainname):
    train=pd.read_csv(filepath+trainname)

    train['label_01']=train.label.apply(lambda x: 0 if x=="normal." else 1)
    train.to_csv("训练.csv")
    #train['new_col'] = list(zip(train.score, train.label_01))
   
    #tac=train.new_col.tolist()
    
    test=pd.read_csv(filepath+testname)

    test['new_col'] = list(zip(test.score, test.label_number))
    tac=test.new_col.tolist()

    tec=test.score.tolist()
    
    p0,p1=VennABERS.ScoresToMultiProbs(tac,tec)
    print(p1.tolist()[9])
    
    test['p0']=p0.tolist()
    test['p1']=p1.tolist()
    test.to_csv('p01.csv')
    
if __name__=='__main__':
    filepath='/home/shaoleshi/民航/数据/NUSW-NB15/'
    testname='测试.csv'
    trainname='训练.csv'
    
    read_csv(filepath,testname,trainname)

