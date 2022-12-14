import NN
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 

#problem 2a forward prop
def questions_2paper_2program():
    x = np.array([1,1,1])
    nn = NN.NeuralNetwork(2,x,3)
    nn.WL1 = np.array(([1,-1,1],[1,-2,2],[1,-3,3]))
    nn.WL2 = np.array(([1,-1,1],[1,-2,2],[1,-3,3]))
    nn.WL3 = np.array([-1,2,-1.5])
    nn.input_vals = np.array([1,1,1])
    g1,g2,g3 = nn.backProp(1)
    return nn,g1,g2,g3

def question_program2b():
    bank_train_data = pd.read_csv("data/train.csv", header=None)
    bank_test_data = pd.read_csv("data/test.csv", header=None)

    data = bank_train_data.to_numpy()
    test =bank_test_data.to_numpy()
    loss={}
    #y0 = 0.05 , d=0.1
    f_yt = lambda t: .05/(1+((.05*t)/.1))
    x = np.array([1,1,1])
    for i in [5,10,25]:
        network = NN.NeuralNetwork(4,x,i)
        network.randomize_w()
        loss_i = network.stochastic_grad_descent(f_yt,data,100)
        loss[i] = loss_i
        train_err,test_err = network.calc_error(data,test)
        print("width = " + str(i) + " train_error = " + str(train_err) + " test_error = " + str(test_err))  
   # f_yt = lambda t: .005/(1+((.005*t)/.1))    
  #  for i in [50,100]:
    #    network = NN.NeuralNetwork(4,x,i)
    #    network.randomize_w()
    #    loss_i = network.stochastic_grad_descent(f_yt,data,100)
    #    loss[i] = loss_i
    #    train_err,test_err = network.calc_error(data,test)
    #    print("width = " + str(i) + " train_error = " + str(train_err) + " test_error = " + str(test_err))

    return loss
def question_program2c():
    bank_train_data = pd.read_csv("data/train.csv", header=None)
    bank_test_data = pd.read_csv("data/test.csv", header=None)

    data = bank_train_data.to_numpy()
    test =bank_test_data.to_numpy()
    loss={}
    #y0 = 0.05 , d=0.1
    f_yt = lambda t: .05/(1+((.05*t)/.1))
    #setup with blank x
    x = np.array([1,1,1])
    for i in [5,10,25]:
        network = NN.NeuralNetwork(4,x,i)
        #leave weights np.zeros
        loss_i = network.stochastic_grad_descent(f_yt,data,100)
        loss[i] = loss_i
        train_err,test_err = network.calc_error(data,test)
        print("width = " + str(i) + " train_error = " + str(train_err) + " test_error = " + str(test_err))  
    
   # f_yt = lambda t: .005/(1+((.005*t)/.1))    
   # for i in [50,100]:
      #  network = NN.NeuralNetwork(4,x,i)
        #leave weights np.zeros
      #  loss_i = network.stochastic_grad_descent(f_yt,data,100)
      #  loss[i] = loss_i
      #  train_err,test_err = network.calc_error(data,test)
       # print("width = " + str(i) + " train_error = " + str(train_err) + " test_error = " + str(test_err))  
    return loss

example,g1,g2,g3 = questions_2paper_2program()
print("forward propagation-----------------------")
print(example.ZL1)
print(example.ZL2)
print(example.y)
print("backward propagation Problem 2A-----------------------")
print("first gradient")
print(g1)
print("second gradient")
print(g2)
print("third gradient")
print(g3)

print("Problem 2B----------------------------------------------------")
resultp2b = question_program2b()
for loss in resultp2b:
    plt.plot(np.linspace(1,100,100),resultp2b[loss])
    plt.show(block=True)

print("Problem 2C----------------------------------------------------")
resultp2c = question_program2c()
for loss in resultp2c:
    plt.plot(np.linspace(1,100,100),resultp2c[loss])
    plt.show(block=True)