# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd
import copy

# ------------------------ 


def crossval_strat(X, Y, n_iterations, iteration):
    
    k=len(X)//n_iterations
    print(type(np.subtract(Y,np.repeat(1,len(Y)))))
    pos=round((np.count_nonzero(Y-1)/len(Y))*k)   #Conservation de la distribution des classes
    neg=k-pos
    res=[]
    
    for i in range(iteration*k,len(Y)):
        if(pos==0 and neg==0):
            break
        if(Y[i]==-1 and neg>0):
            neg=neg-1
            res.append(i)
        elif(Y[i]==1 and pos>0):
            pos=pos-1
            res.append(i)
    
    if(pos!=neg and neg !=0):
        for i in range(iteration*k):
            if(pos==0 and neg==0):
                break
            if(Y[i]==-1 and neg>0):
                neg=neg-1
                res.append(i)
            elif(Y[i]==1 and pos>0):
                pos=pos-1
                res.append(i)
    arange=np.arange(len(X))
    arange=np.setdiff1d(arange,np.array(res))               # Liste de 0 à len(X) ne contenant pas les élements de res
    Xapp=X[arange]
    Yapp=Y[arange]
    Xtest=X[res]
    Ytest=Y[res]
    return Xapp, Yapp, Xtest, Ytest
    
    
def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    arrL = np.array(L)
    moy = arrL.sum()/len(arrL)
    e_t = np.std(arrL, dtype = np.float64)
    return moy,e_t

def cout_perceptron(w,X,Y):
    N = X.shape[0]
    C=0
    for i in range(0,N):
        f_xi = np.dot(w,X[i])
        yi = Y[i]
        in_sum = 1-f_xi*yi
        if in_sum>0:
            C+=in_sum
    return C   

def validation_croisee(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    X, Y = DS   
    perf = []
    
    for i in range(nb_iter):
        newC = copy.deepcopy(C)
        desc_train,label_train,desc_test,label_test=crossval_strat(X,Y,nb_iter,i)
        newC.train(desc_train,label_train)
        acc_i=newC.accuracy(desc_test,label_test)
        perf.append(acc_i)
    (perf_moy, perf_sd) = analyse_perfs(perf)
    return (perf, perf_moy, perf_sd)

# ------------------------ 
