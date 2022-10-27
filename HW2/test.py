import adaboost as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
#train on tennis

def test_bank(t):
    train_data = pd.read_csv("HW2/Data/bank-2/train.csv", header=None)
    test_data = pd.read_csv("HW2/Data/bank-2/test.csv", header=None)
    feat={"age":0, "job":1, "marital":2, "education":3,"default":4, "balance":5, 
          "housing":6,"loan":7,"contact":8,"day":9,"month":10,"duration":11,
          "campaign":12,"pdays":13,"previous":14,"poutcome":15 }
   
    l_y = ["yes"]
    l_n = ["no"]
   
    clf = ad.Adaboost()
    clf.fit(train_data, feat, l_y, l_n,t)
    y_pred = clf.predict(test_data,feat,l_y)
    n_samples = len(test_data)
    y_test = np.ones(n_samples)
 
    acc = accuracy(y_test, y_pred)
    return acc
acc = np.zeros(10)
for t in range(1,10):
  acc[t] = test_bank(t)

xpoints = np.array([1,10])
plt.plot(xpoints,acc)
plt.xlabel('iterations') 
plt.ylabel('accuracy') 
plt.title("accuracy improvment")
plt.show()
print("Accuracy:", acc)