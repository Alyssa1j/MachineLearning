import pandas as pd
import numpy as np
import SVM

bank_train_data = pd.read_csv("data/bank-note/train.csv", header=None)
bank_test_data = pd.read_csv("data/bank-note/test.csv", header=None)

data = bank_train_data.to_numpy()
test =bank_test_data.to_numpy()

c= [100/873,500/873,700/873]
#Part 2 problem 2------------------------------------------
#2a)

f_yt = lambda r,a,t: r/(1+((r*t)/a))
print("My 2A results.............")
w1 = SVM.svm_stochastic_subgrad(data,t=100, r=0.1, a=0.1, C=c[0], fyt=f_yt)
w2 = SVM.svm_stochastic_subgrad(data,t=100, r=0.1, a=0.1, C=c[1],fyt =f_yt)
w3 = SVM.svm_stochastic_subgrad(data,t=100, r=0.1, a=0.1, C=c[2],fyt =f_yt)

w1_e = SVM.error_rate(data, w1)
w2_e = SVM.error_rate(data, w2)
w3_e = SVM.error_rate(data, w3)
print("-->Training results--------------")
print("c1: ",w1, "error: ", w1_e)
print("c2: ",w2, "error: ", w2_e)
print("c3: ",w3, "error: ", w3_e)

w1 = SVM.svm_stochastic_subgrad(test,t=100, r=0.1, a=0.1, C=c[0], fyt=f_yt)
w2 = SVM.svm_stochastic_subgrad(test,t=100, r=0.1, a=0.1, C=c[1], fyt=f_yt)
w3 = SVM.svm_stochastic_subgrad(test,t=100, r=0.1, a=0.1, C=c[2], fyt=f_yt)

w1_e = SVM.error_rate(test, w1)
w2_e = SVM.error_rate(test, w2)
w3_e = SVM.error_rate(test, w3)

print("-->Test results--------------")
print("c1: ",w1, "error: ", w1_e)
print("c2: ",w2, "error: ", w2_e)
print("c3: ",w3, "error: ", w3_e)

#2b)
f_yt = lambda r,a,t: r/(1+t)
print("My 2b results.............")
w1 = SVM.svm_stochastic_subgrad(data,t=100, r=0.1, a=0.1, C=c[0], fyt=f_yt)
w2 = SVM.svm_stochastic_subgrad(data,t=100, r=0.1, a=0.1, C=c[1],fyt =f_yt)
w3 = SVM.svm_stochastic_subgrad(data,t=100, r=0.1, a=0.1, C=c[2],fyt =f_yt)

w1_e = SVM.error_rate(data, w1)
w2_e = SVM.error_rate(data, w2)
w3_e = SVM.error_rate(data, w3)
print("-->Training results--------------")
print("c1: ",w1, "error: ", w1_e)
print("c2: ",w2, "error: ", w2_e)
print("c3: ",w3, "error: ", w3_e)

w1_e = SVM.error_rate(test, w1)
w2_e = SVM.error_rate(test, w2)
w3_e = SVM.error_rate(test, w3)

print("-->Test results--------------")
print("c1: ",w1, "error: ", w1_e)
print("c2: ",w2, "error: ", w2_e)
print("c3: ",w3, "error: ", w3_e)

#Part 2 problem 3---------------------------------------------------
'''
print("My 3A results--------------------------------------")
result = SVM.svm_dual(data,c)
for r in result:
    print("c= ", r[0], "w= ", r[1], "b= ", r[2] )

#calculate error rate by appending bias term to weights and utilize error_rate function

w1_e = SVM.error_rate(data, np.append(result[0][1], result[0][2]))
w2_e = SVM.error_rate(data, np.append(result[1][1], result[1][2]))
w3_e = SVM.error_rate(data, np.append(result[2][1], result[2][2]))
print("-->Training results--------------")
print("c1: ",c[0], "error: ", w1_e)
print("c2: ",c[1], "error: ", w2_e)
print("c3: ",c[2], "error: ", w3_e)


w1_e = SVM.error_rate(test, np.append(result[0][1], result[0][2]))
w2_e = SVM.error_rate(test, np.append(result[1][1], result[1][2]))
w3_e = SVM.error_rate(test, np.append(result[2][1], result[2][2]))

print("-->Test results--------------")
print("c1: ",c[0], "error: ", w1_e)
print("c2: ",c[1], "error: ", w2_e)
print("c3: ",c[2], "error: ", w3_e)

'''

#part 3b gaussian kernel in dcual form-----------------------------
'''
gamma = [0.1,0.5,1,5,100]

results, support_vectors = SVM.svm_dual_GausKernel(data,test,c,gamma)

#returns list of [c_val,gamma, bias on train, bias on test, train error rate, test error rate]
print("Results for 3b----------------------------")
for r in results:
    print("c= ", r[0],"g=",r[1],"b_train= ", r[2], "b_test= ", r[3], "train error = ", r[4],"test error = ", r[5])
print("Results for 3c----------------------------")
count =0
c_old = support_vectors[0][0]
for s in support_vectors:
    c_new=s[0]
    if c_new == c_old:
        count+=1
    else:
        print("Total count: ", count)
        count =1
        c_old = c_new

    print("c= ", s[0],"g=",s[1], "sv= ", s[2])

'''
