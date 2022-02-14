
import numpy as np
import math
import pandas as pd
import random as r
import csv
import time
import os
import sys

from numpy.random import RandomState

import gurobipy as gp
from gurobipy import GRB


from sklearn.ensemble import BaggingClassifier
from sklearn import svm
from Parameters import *


def printStatus(stat):
    
    if stat  == GRB.OPTIMAL:
        pass
        #print('Model is optimal')
    elif stat  == GRB.INF_OR_UNBD:
        print('Model  is  infeasible  or  unbounded')
    elif stat  == GRB.INFEASIBLE:
        print('Model  is  infeasible')
    elif stat  == GRB.UNBOUNDED:
        print('Model  is  unbounded')
    else:
        print('Optimization  ended  with  status ' + str(stat))

def trainSVM(X,y):
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)
    
    return clf

def trainSVMEnsemble(X,y,k):
    clf = BaggingClassifier(base_estimator=svm.SVC(kernel='linear'),n_estimators=k, random_state=32)
    clf.fit(X, y)
    
    return clf

def getOptimalDeltaAverage(x,y,w,b,k,r,norm):
    #print("Adversarial Problem Calculation")
    ip = gp.Model("AdversarialProblem_SVM")
    ip.setParam("OutputFlag",0)
    ip.setParam("TimeLimit", timeLimit)
    #ip.setParam('MIPGap', MIPGap)
    
    n = x.shape[0]
    
    if norm == 1:
        v = ip.addVars(k,vtype=GRB.CONTINUOUS, lb = 0, ub=1, name="v")
        mu = ip.addVars(n,vtype=GRB.CONTINUOUS, name="mu")
        z = ip.addVars(n,vtype=GRB.BINARY, name="z")
        u = ip.addVars(n,vtype=GRB.BINARY, name="u")
        
        lhs = ""
        for l in range(n):
            lhs = lhs + r * u[l]*mu[l]
        for i in range(k):
            coeff = 1 - y*(np.dot(w[i,:],x)+b[i])
            lhs = lhs + coeff * v[i]
        ip.setObjective(lhs, GRB.MAXIMIZE)
        
        #Constraints
        lhs = ""
        for l in range(n):
            lhs = lhs + 1 * u[l]
        ip.addConstr(lhs, sense=GRB.EQUAL, rhs=1)
        
        for l in range(n):
            lhs= 1 * mu[l] - k*z[l]
            for i in range(k):
                lhs = lhs + y*w[i,l]*v[i]

            ip.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=0)
        for l in range(n):
            lhs= 1 * mu[l] + k*z[l]
            for i in range(k):
                lhs = lhs - y*w[i,l]*v[i]

            ip.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=k)
            
        
    elif norm>2:
        v = ip.addVars(k,vtype=GRB.CONTINUOUS, lb = 0, ub=1, name="v")
        mu = ip.addVars(n,vtype=GRB.CONTINUOUS, name="mu")
        xi = ip.addVars(n,vtype=GRB.BINARY, name="xi")
        
        
        lhs = ""
        for l in range(n):
            lhs = lhs +  r * mu[l]
        for i in range(k):
            coeff = 1 - y*(np.dot(w[i,:],x)+b[i])
            lhs = lhs + coeff * v[i]
        ip.setObjective(lhs, GRB.MAXIMIZE)
        
        
        #Constraints
        for l in range(n):
            lhs= - 1 * mu[l]
            for i in range(k):
                lhs = lhs - w[i,l]*v[i]
                lhs = lhs + 2*w[i,l]*xi[l]*v[i]
            ip.addConstr(lhs, sense=GRB.EQUAL, rhs=0)
            
            lhs= - k * xi[l]
            for i in range(k):
                lhs = lhs + w[i,l]*v[i]
            ip.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=0)
            
            lhs= k * xi[l]
            for i in range(k):
                lhs = lhs - w[i,l]*v[i]
            ip.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=k)
        
        
        
    ip.optimize()
    #ip.write("RO-SVM.lp")  

    printStatus(ip.status)
    
    delta_ret = np.zeros(n)
    ATv = 0
    for i in range(k):
        ATv = ATv - y*v[i].x*w[i,:]
        
    if norm==1:
        if np.amax(ATv,axis=0)==0 and np.amin(ATv,axis=0)==0:
            maxIndex = np.random.randint(n)
            delta_ret[maxIndex]=(-1 + 2*np.random.randint(0,2))*r
        else:
            maxIndex = np.argmax(np.absolute(ATv),axis=0)
            delta_ret[maxIndex]=np.sign(ATv[maxIndex])*r

        
    elif norm>2:
        for l in range(n):
            if ATv[l]>0:
                delta_ret[l] = r
            elif ATv[l]==0:
                delta_ret[l] = (-1 + 2*np.random.randint(0,2))*r
            else:
                delta_ret[l] = -r
        
    return delta_ret

    
def getWorstCaseAttackHeuristic(x,y,w,b,k,r,norm):
    
    #Calculate weights for each hyperplane
    lam = np.zeros(k,dtype=float)
    
    for i in range(k):
        lam[i] = max(0,1+y*(np.dot(w[i,:],x)+b[i]))
    
    if np.sum(lam)>0:    
        lam = lam / np.sum(lam)
        
    delta = 0
    for i in range(k):
        delta = delta + lam[i]*w[i,:]
    
    if norm==2:
        if np.linalg.norm(delta,2)>0:
            delta = delta / np.linalg.norm(delta,2)
    elif norm > 2:
        delta = r * np.sign(delta)
    elif norm==1:
        print("Not implemented")
        
        
    return delta
        

def getWorstCaseAttackNumHyperplanes(x,y,w,b,k,r,norm):
    #print("Adversarial Problem Calculation")
    ip = gp.Model("WCAttacks_SVM")
    ip.setParam("OutputFlag",0)
    ip.setParam("TimeLimit", timeLimit)
    #ip.setParam('MIPGap', MIPGap)
    
    n = x.shape[0]
    
    d = ip.addVars(n,vtype=GRB.CONTINUOUS, lb = -r, ub=r, name="d")
    z = ip.addVars(k,vtype=GRB.BINARY, name="z")
    
    
    lhs = ""
    for i in range(k):
        lhs = lhs - 1 * z[i]
    ip.setObjective(lhs, GRB.MAXIMIZE)
    
    
    #Constraints
    for i in range(k):
        M = y*(np.dot(w[i,:],x)+b[i])+n*r
        lhs= - M * z[i]
        for l in range(n):
            lhs = lhs + y*w[i,l]*d[l]
        ip.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=-y*(np.dot(w[i,:],x)+b[i]))
        
    if norm > 2:
        pass
    elif norm == 2:
        lhs = ""
        for i in range(n):
            lhs = lhs + d[i]*d[i]
        ip.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=r*r)
        

    ip.optimize()

    printStatus(ip.status)
    
    delta_ret = np.zeros(n)
    for l in range(n):
        delta_ret[l] = d[l].x

        
    return delta_ret


def iterativeHeuristic(X,y,norm, r,k, worstCaseType):
    n = X.shape[1]
    m = X.shape[0]
    w = np.zeros((k,n),dtype=float)
    b = np.zeros(k,dtype=float)
    
    #Set first solution to robust SVM solution
    w_rob,b_rob = trainRobustSVM(X,y,norm,r)
    w[0,:] = w_rob[:]
    b[0] = b_rob
    
    timeWorstCaseCalculations = 0
    numWorstCaseCalculations = 0
    for i in range(1,k):
        print("Iteration:",i)
        attackVector = np.ones(X.shape)

        for j in range(m):
            #Calculate Hyperplanes which are close enough such that they are vulnerable
            vulnerableHyperplanes = []
            for l in range(i):
                if norm > 2:
                    if y[j]==-1:
                        attack = r * np.sign(w[l,:])
                    else:
                        attack = -r * np.sign(w[l,:])
                        
                    if y[j]*(np.dot(w[l,:],X[j,:]+attack)+b[l])>eps:
                        pass
                    else:
                        vulnerableHyperplanes.append(l)
                elif norm==2:
                    if y[j]==-1:
                        attack = (r*w[l,:])/np.linalg.norm(w[l,:])
                    else:
                        attack = (-r*w[l,:])/np.linalg.norm(w[l,:])
                        
                    if y[j]*(np.dot(w[l,:],X[j,:]+attack)+b[l])>eps:
                        pass
                    else:
                        vulnerableHyperplanes.append(l)
            
            if len(vulnerableHyperplanes)>0:
                start = time.time()
                if worstCaseType == "Ens-E":        
                    attackVector[j,:] = getWorstCaseAttackNumHyperplanes(X[j,:],y[j],w[vulnerableHyperplanes,:],b[vulnerableHyperplanes],len(vulnerableHyperplanes),r,norm)
                elif worstCaseType == "Ens-R":
                    attackVector[j,:] = getOptimalDeltaAverage(X[j,:],y[j],w[vulnerableHyperplanes,:],b[vulnerableHyperplanes],len(vulnerableHyperplanes),r,norm)
                elif worstCaseType == "Ens-H":
                    attackVector[j,:] = getWorstCaseAttackHeuristic(X[j,:],y[j],w[vulnerableHyperplanes,:],b[vulnerableHyperplanes],len(vulnerableHyperplanes),r,norm)
                end = time.time()
                timeWorstCaseCalculations = timeWorstCaseCalculations + end - start
                numWorstCaseCalculations+=1
            else:
                attackVector[j,:] = attack

        X_new = X + attackVector
        
        #define weights for each data-point
        data_weights = np.ones(X_new.shape[0])
        predictionError = 0
        for j in range(X_new.shape[0]):
            val = 0
            for l in range(i):
                val = val + np.sign(np.dot(w[l,:],X_new[j,:]) + b[l])
            if y[j]*val<=0:
                predictionError+=1
                
            val = y[j]*val + i
            data_weights[j] = 1.0 / (1+val)
            
            
        clf = svm.SVC(kernel='linear')
        clf.fit(X_new, y, data_weights)
        w[i,:] = clf.coef_
        b[i] = clf.intercept_
        
        if numWorstCaseCalculations==0:
            avgCalcTime = 0
        else:
            avgCalcTime = timeWorstCaseCalculations / numWorstCaseCalculations
        
    return w, b, avgCalcTime
    


def trainRobustSVM(X,y,norm, r):
    
     # Create a new model
    ip = gp.Model("RO-SVM")
    ip.setParam("OutputFlag",0)
    ip.setParam("TimeLimit", timeLimit)
    #ip.setParam('MIPGap', MIPGap)
    
    n = X.shape[1]
    m = X.shape[0]
    
    w = ip.addVars(n,vtype=GRB.CONTINUOUS, lb = -GRB.INFINITY, name="w")
    b = ip.addVar(vtype=GRB.CONTINUOUS, lb = -GRB.INFINITY, name="b")
    
    xi = ip.addVars(m, vtype=GRB.CONTINUOUS, name="xi")
    
    lhs = ""
    for j in range(m):
        lhs = lhs + 1 * xi[j]
    ip.setObjective(lhs, GRB.MINIMIZE)
    
    
    if norm == 2:
        mu = ip.addVar(vtype=GRB.CONTINUOUS, name="mu")
        for j in range(m):
            lhs = r * mu - 1 * xi[j] - y[j]*b
            for i in range(n):
                lhs = lhs - y[j]*X[j,i]*w[i]
    
            ip.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=-1)
            
        lhs = - 1 * mu*mu
        for i in range(n):
            lhs = lhs + w[i]*w[i]
        ip.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=0)
    elif norm > 2:
        mu = ip.addVars(n, vtype=GRB.CONTINUOUS, name="mu")
        for j in range(m):
            lhs = - 1 * xi[j] - y[j]*b
            for i in range(n):
                lhs = lhs - y[j]*X[j,i]*w[i]
                lhs = lhs + r*mu[i]
    
            ip.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=-1)
            
        for i in range(n):
            lhs = 1*w[i] - 1*mu[i]
            ip.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=0)
            
            lhs = - 1 * w[i] - 1* mu[i]
            ip.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=0)
            
            
        
    
    ip.optimize()
    #ip.write("RO-SVM.lp")  

    printStatus(ip.status)
    
    w_ret = np.zeros(n)
    b_ret = b.x
    for i in range(n):
        w_ret[i] = w[i].x
        
    return w_ret, b_ret

def predictRobustSVM(w,b,X):
    y_pred = np.zeros(X.shape[0])
    
    for i in range(X.shape[0]):
        y_pred[i] = np.sign(np.dot(w,X[i,:]) + b)
        if y_pred[i]==0:
            y_pred[i] = 1
        
    return y_pred

def predictMultiRobustSVM(w,b,X):
    y_pred = np.zeros(X.shape[0])
    
    for j in range(X.shape[0]):
        summe = 0
        for i in range(w.shape[0]):
            summe = summe + np.sign(np.dot(w[i,:],X[j,:]) + b[i])
        if summe>=0:
            y_pred[j] = 1
        else:
            y_pred[j] = -1
        
    return y_pred