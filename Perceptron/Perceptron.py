import numpy as np
from matplotlib import pyplot as plt


#Standard perceptron
def perceptron_std(X, Y, t,r):
    weights = np.zeros(len(X[0]))
    for epoch in range(0,t):
        np.random.shuffle(X)
        for i, x in enumerate(X):
            if (np.dot(X[i], weights)*Y[i]) <= 0:
                weights = weights + r*X[i]*Y[i]
    return weights

#Standard perceptron prediction
def std_pred(X,Y,w):
    preds =[]
    errorCount=0
    c=0
    for i, x in enumerate(X):
        pred = np.sign(np.dot(X[i], w))
        preds.append(pred)
        if(Y[i] != pred):
            errorCount+=pred
        else:
            c+=1

    p = errorCount / len(preds)
    print("Avg Prediction Error: ", p)
    return preds,c

#Standard perceptron with training plot
def perceptron_std_plot(X, Y, t, r):
    w = np.zeros(len(X[0]))
    errors = []

    for epoch in range(t):
        total_error = 0
        np.random.shuffle(X)
        for i, x in enumerate(X):
            if (np.dot(X[i], w)*Y[i]) <= 0:
                total_error += (np.dot(X[i], w)*Y[i])
                w = w + r*X[i]*Y[i]
        errors.append(total_error)
    #avg = sum(errors)/len(errors)

    #print("Average Error: ",avg)    
    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.show()
    return w



#voted perceptron ----------------------------------------------------------------------------------------
def perceptron_vtd(X, Y, t,r):
    w = np.zeros(len(X[0]))
    m=0
    c=0
    preds=[]
    W=[]
    for epoch in range(0,t):
        for i, x in enumerate(X):
            if (np.dot(X[i], w)*Y[i]) <= 0:
                #add m,wm,cm to list.
                #m is a weight counter
                preds.append((w,c))
                W.append(w)
                #calculate w_m+1
                w = w + r*X[i]*Y[i]
                m +=1
                c=1
            else:
                c=c+1    
    return preds,W

#voted perceptron prediction
def  predict_vtd(WC,X, y):
    predictions = []
    errorSum=0
    c=0
    for x in range(len(X)):
        s  = 0
        for i in range(len(WC)):
            s = s + WC[i][1]*np.sign(np.dot(WC[i][0],X[x]))
        predictions.append(np.sign(s))
        if(y[x] != np.sign(s)):
            errorSum += np.sign(s)
        else:
            c+=1

    p = errorSum / len(y)
   # print("ErrorSum: ", errorSum, "total: ", len(y), "C:", c)
    print("Average Test Error: ", p)
    return predictions,c


#averaged perceptron------------------------------------------------------------------------------
def perceptron_avg(X, Y, t,r):
    weights = np.zeros(len(X[0]))
    avg = np.zeros(len(X[0]))
    for epoch in range(0,t):
      #  np.random.shuffle(X)
        for i, x in enumerate(X):
            if (np.dot(X[i], weights)*Y[i]) <= 0:
                weights = weights + r*X[i]*Y[i]
            avg = avg + weights
    return avg,weights

#Standard perceptron prediction
def avg_pred(X,Y,a):
    preds =[]
    errorCount=0
    c=0
    for i, x in enumerate(X):
        pred = np.sign(np.dot(X[i], a))
        preds.append(pred)
        if(Y[i] != pred):
            errorCount+=pred
        else:
            c+=1

    p = errorCount / len(preds)
    print("Prediction Error: ", p)
    return preds,c