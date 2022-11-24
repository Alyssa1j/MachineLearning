import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import optimize
#SVM with stochastic subgradient descent
def svm_stochastic_subgrad(data, t,r,a,C, fyt):
    N = len(data)
    weights = np.zeros(5)
    for epoch in range(0,t):
        np.random.shuffle(data)
        lt = fyt(r,a,epoch)
        for i, d in enumerate(data):
            x = d[0:5]
            x[-1]=1
            y = np.copy(d[4])
            if y <=0:
                y=-1
            if (np.dot(x, weights)*y) <= 1:
                w0 = weights.copy()
                w0[-1] =0
                weights = weights -(lt * w0)+(lt*C*N*x*y)
            else:
                weights = (1-lt)*weights
    return weights

#error rate calculation
def error_rate(data,weights):
    count =0
    for i,d in enumerate(data):
        x=d[0:5]
        x[-1]=1
        y=d[-1]
        if y <=0:
            y=-1
        if (np.dot(x, weights)*y) <= 1:
            count +=1

    return count/len(data)

#svm_duality equation, returns an array of vectors containing [c,w,b] from which you can use to calculate the error
#utilizes some references for help with optimize.minimize function from:
###https://tonio73.github.io/data-science/classification/ClassificationSVM.html
###https://stackoverflow.com/questions/61022308/converting-double-for-loop-to-numpy-linear-algebra-for-dual-form-svm
def svm_dual(data,c_vec):
    x_d = data[:,0:4]
    y_d = np.where(data[:,4] <=0,-1,1)
    cons = ({'type': 'eq', 'fun': lambda x,y: np.dot(x,y),'args': [y_d]})
    results = []

    for c in c_vec:
        bnds = optimize.Bounds(0,c)
        optResult = optimize.minimize(lang_dual,np.zeros(872),data,method='SLSQP',bounds=bnds,constraints=cons)
        w,b = solve_dual(optResult.x, x_d, y_d)
        results.append([c,w,b])
        
    return results

def lang_dual(a,data):
    xi = data[:,0:4]
    yi = np.where(data[:,4] <=0,-1,1)
    #optimized the sumation using a reference from stack overflow
    vectorSum = 0.5 * np.einsum("i,j,i,j,ix,jx->",yi,yi,a,a,xi,xi,optimize="optimal")
    return vectorSum - np.sum(a)

def solve_dual(a,x,y):
    ay_vec = a*y
    ay_vec.reshape(-1,1)
    w = np.sum(ay_vec[:,None] * x, axis=0)
    #divides by length of y_j given on slide 51 from svm-dual-kernel-tricks     
    b = np.sum(y - np.dot(x,w))/len(y)
    return w,b
