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
            x = np.copy(d)
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
        x=np.copy(d)
        x[-1]=1
        y=np.copy(d[4])
        if y <=0:
            y=-1
        if ((x@weights)*y) <=0:
            count +=1

    return count/len(data)

#svm_duality equation, returns an array of vectors containing [c,w,b] from which you can use to calculate the error
#utilizes some references for help with optimize.minimize function from:
###https://tonio73.github.io/data-science/classification/ClassificationSVM.html
def svm_dual(data,c_vec):
    x_d = data[:,0:4]
    y_d = np.where(data[:,4] <=0,-1,1)
    cons = ({'type': 'eq', 'fun': lambda x,y: np.dot(x,y),'args': [y_d]})
    results = []
    N = len(data)
    for c in c_vec:
        bnds = optimize.Bounds(0,c)
        optResult = optimize.minimize(lang_dual,np.zeros(N),data,method='SLSQP',bounds=bnds,constraints=cons)
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

def lang_dual_kernel(a,data,gamma):
    xi = data[:,0:4]
    yi = np.where(data[:,4] <=0,-1,1)
    norm= np.einsum('ij,ij->i',xi,xi)
    k = np.exp(-(norm[:,None] + norm[None,:] - 2 * np.dot(xi, xi.T))/gamma)
    vectorSum = 0.5 * np.einsum("i,j,i,j,ij->",yi,yi,a,a,k,optimize="optimal")
    return vectorSum - np.sum(a)


#dual gaussian kernel
#returns list of [c_val,gamma, bias on train, bias on test, train error rate, test error rate]
#return list of support vectors, may contain duplicates [c_val,gamma,support_vec]
def svm_dual_GausKernel(dataTrain, dataTest,c_vec,gamma):
    x_d = dataTrain[:,0:4]
    y_d = np.where(dataTrain[:,4] <=0,-1,1)
    y_test = np.where(dataTest[:,-1] == 0,-1,1)
    cons = ({'type': 'eq', 'fun': lambda x,y: np.dot(x,y),'args': [y_d]})
    results = []
    support_vectors=[]
    N = len(dataTrain)
    print("starting loop")
    for c in c_vec:
        for g in gamma:
            print("c= ", str(c),"g= ", str(g))
            bnds = optimize.Bounds(0,c)
            optResult = optimize.minimize(lang_dual_kernel,np.zeros(N),(dataTrain,g),method='SLSQP',bounds=bnds,constraints=cons)
            support_vec =  find_supportV(optResult.x)
            support_vectors.append([c,g,support_vec])
            vTrain = kernel_pred(optResult.x, g, x_d, y_d, x_d)
            vTest =  kernel_pred(optResult.x, g, x_d, y_d, dataTest[:,0:4])
            b_train = np.mean(y_d-vTrain)
            b_test = np.mean(y_test-vTest)
            dataTrain_error=kernel_error_rate(vTrain,b_train,y_d)
            test_error=kernel_error_rate(vTest,b_test,y_test)
            results.append([c,g,b_train, b_test,dataTrain_error,test_error])
            print("result: ", [c,g,b_train, b_test,dataTrain_error,test_error])
            
    return results, support_vectors

#get kernel prediction, 
#a=alpha vector
#g=gamma
#x_d, x axis data
#y_d, y axis data
#x_repeat, used to create z-axis data
#returns-pred_vector
def kernel_pred(a,g,x_d,y_d,x_repeat):
    pred_vector = np.empty(len(x_repeat[:,0]))
    for i in range(len(x_repeat)):
        z = np.repeat(x_repeat[i][np.newaxis,:],x_d.shape[0],axis=0)
        kernal = np.exp(-np.square(np.linalg.norm(x_d  - z,axis=1))/g)
        pred_vector[i] = np.sum(a*y_d*kernal)
    return pred_vector

def find_supportV(a):
    support_vec_dict = {}
    for i,a_i in enumerate(a):
        if a_i == 0:
            support_vec_dict[i] = 1
    return support_vec_dict


def kernel_error_rate(v,b,y):
    count =0
    for i,yi in enumerate(y):
        if yi*(v[i] + b) <=0:
            count +=1
    return count/len(y)