
import numpy as np
import math
import pandas as pd
import random as r
import time
import os
import sys

from numpy.random import RandomState

import gurobipy as gp
from gurobipy import GRB

from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from sklearn import svm
from Parameters import *
import Functions as f


epsDefend = 0.01
normDefend = 2
normAttack = 2  #inf-norm: value greater than 2
attackType = "worstcase" #random worstcase
scaler = "Standardize"  #MinMax, Standardize
numTrainTestSplits = 1

k_range = [5]  # values of k for which calculations will be performed

model = "MROSVM" # SVMEnsemble, ROSVM, MROSVM
worstCaseType = "Ens-H"  # exact adversarial: Ens-E, relaxed adversarial: Ens-R (only for l1 and linf-norm), heuristic adversarial: Ens-H

dataset = "breast_cancer"
fixed_digit = 3  #only for Digits dataset: fixed digit assigned to class 1 in Digits dataset
fileName = dataset + "_" + model + "_" + worstCaseType + ".csv"

randVals = RandomState(32)

if dataset=="breast_cancer":
    targetColumn = 9
    data = np.genfromtxt('Data/breast-cancer-wisconsin.csv',dtype = float, delimiter=';')
    y=data[:,targetColumn]
    X=data[:,np.delete(np.arange(0,data.shape[1],1),targetColumn)]
elif dataset == "digits":
    X,y = load_digits(return_X_y=True)
    X = np.array(X)
    y=np.array(y)
    y[y!=fixed_digit]=0
    y[y==fixed_digit]=1
elif dataset=="randGaussian":
    clusterSize = 100
    dim = 5
    mean = np.ones(dim)
    sigma = np.ones(dim)
    Sigma = np.diagflat(sigma)
        
    U1 = randVals.multivariate_normal(mean, Sigma, clusterSize)
    y1=np.ones(clusterSize, dtype=int)
    
    U2 = randVals.multivariate_normal(-mean, Sigma, clusterSize)
    y2=np.zeros(clusterSize, dtype=int)

    X = np.append(U1,U2,axis=0)
    y = np.append(y1,y2)

       
n=X.shape[1]
y = -1 + 2*y
timeWC = 0

for randShuffle in range(numTrainTestSplits):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=fracTest, random_state=32, shuffle=True)
        
    if scaler=="Standardize":
        mu = np.mean(X_train,axis=0)
        std = np.std(X_train,axis=0)
        for i in range(std.shape[0]):
            if std[i] == 0:
                std[i] = 1
        for i in range(X_train.shape[0]):
            X_train[i,:] = (X_train[i,:]-mu)/std
        for i in range(X_test.shape[0]):
            X_test[i,:] = (X_test[i,:]-mu)/std
    elif scaler == "MinMax":
        scalerMM = MinMaxScaler()
        X_train = scalerMM.fit_transform(X_train)
        X_test = scalerMM.fit_transform(X_test)
    
    print("X_train.shape:",X_train.shape)
    print("y_train.shape:",y_train.shape)
    print("X_test.shape",X_test.shape)
    print("y_test.shape",y_test.shape)
    
    for k in k_range:
        if model=="SVM" or model=="ROSVM":
            numSolutions=1
        else:
            numSolutions = k
        
        print("\nStart model:",model)
        start = time.time()
        if model=="SVMEnsemble":
            svm_trained = f.trainSVMEnsemble(X_train,y_train, k)
            models = svm_trained.estimators_
            w = np.zeros((k,n),dtype=float)
            b = np.zeros(k)
            for i in range(k):
                w[i,:] = models[i].coef_
                b[i] = models[i].intercept_       
        elif model == "ROSVM":
            w,b = f.trainRobustSVM(X_train,y_train,normDefend,epsDefend)
        elif model == "MROSVM":
            w,b, timeWC = f.iterativeHeuristic(X_train,y_train,normDefend,epsDefend,k,worstCaseType)
        end = time.time()
        
        runtime = end - start
        
        print(w)
        print(b)
            
        for epsAttack in [0.0,0.1,0.2,0.3,0.4,0.5,0.75,1.0,1.25,1.5,1.75,2.0]:
            
            if attackType == "worstcase":
                if normAttack > 2:
                    attackVector = np.ones(X_test.shape)
                    for i in range(X_test.shape[0]):
                        if model == "SVM" or model == "ROSVM":
                            if y_test[i]==-1:
                                attackVector[i,:] = epsAttack * np.sign(w)
                            else:
                                attackVector[i,:] = -epsAttack * np.sign(w)
                        else:
                            attackVector[i,:] = f.getWorstCaseAttackNumHyperplanes(X_test[i,:],y_test[i],w,b,k,epsAttack,normAttack)
                if normAttack == 2:
                    attackVector = np.zeros(X_test.shape)
                    for i in range(X_test.shape[0]):
                        if model == "SVM" or model == "ROSVM":
                            if y_test[i]==-1:
                                attackVector[i,:] = (epsAttack*w)/np.linalg.norm(w)
                            else:
                                attackVector[i,:] = (-epsAttack*w)/np.linalg.norm(w)
                        else:
                            attackVector[i,:] = f.getWorstCaseAttackNumHyperplanes(X_test[i,:],y_test[i],w,b,k,epsAttack,normAttack)
    
            
            if model=="SVMEnsemble":
                y_pred = f.predictMultiRobustSVM(w,b,X_test+attackVector)
            elif model == "ROSVM":
                y_pred = f.predictRobustSVM(w,b,X_test+attackVector)
            elif model == "MROSVM":
                y_pred = f.predictMultiRobustSVM(w,b,X_test+attackVector)
                
            accuracy = accuracy_score(y_test,y_pred)
    
            row = [normDefend,normAttack,numSolutions,epsDefend, epsAttack, accuracy, runtime, timeWC]
            df = pd.DataFrame(row)
            df = df.transpose()
            df.to_csv(fileName, sep=';', mode='a',header=False, index=False)



