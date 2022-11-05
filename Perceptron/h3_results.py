import Perceptron
import pandas as pd
import numpy as np
bank_train_data = pd.read_csv("data/bank-note/train.csv", header=None)
bank_test_data = pd.read_csv("data/bank-note/test.csv", header=None)

x = bank_train_data.iloc[:,0:5].to_numpy()
y = bank_train_data.iloc[:,4].to_numpy()
xtest =bank_test_data.iloc[:,0:5].to_numpy()
ytest=bank_test_data.iloc[:,4].to_numpy()

#2a)
print("2A results.............")
w = Perceptron.perceptron_std_plot(xtest,ytest,t=10, r=0.01)
print("Learned Weight vector: ",w)
p,c = Perceptron.std_pred(xtest,ytest,w)
print("correctly predicted test: ", c)
print("2B results.............")
print("trained data: ")
p,w = Perceptron.perceptron_vtd(x,y,t=10, r=0.01)
print("Learned Weight vectors size: ",len(p))
print("First 10: ", p[0:10])
predictions,c = Perceptron.predict_vtd(p,x, y)
print("Correctly predicted training examples: ",c)
print("Test data.....")
predictions,c = Perceptron.predict_vtd(p,xtest, ytest)
print("Correctly predicted test examples: ",c)

print("2C results.............")
a,w = Perceptron.perceptron_avg(x,y,t=10, r=0.01)
print("Learned Weight Vector: ",w)
print("Learned Average Vector: ", a)
p,c = Perceptron.avg_pred(xtest,ytest,a)
print("Correctly predicted test examples: ",c)
