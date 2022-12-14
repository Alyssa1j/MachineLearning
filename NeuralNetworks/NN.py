import numpy as np
class NeuralNetwork:
    '''
    3 layer nerual network

    '''
    def __init__(self, numInputs,x,width):
        self.WL1 = np.zeros((numInputs+1,width))
        self.WL2 =np.zeros((width,width))
        self.WL3=np.zeros(width)
        self.ZL1 = np.zeros(width)
        self.ZL2 = np.zeros(width)
        #layers extra inputs are set to 1 described in assignment
        self.ZL1[0] =1
        self.ZL2[0]=1
        self.w = width
        self.x =x
        self.numInputs = numInputs+1
        self.y=0


    def sigmoid(self,x):
        return 1.0/(1.0 + np.exp(-x))

    def forwardProp(self):
        #calculate first layer of z values
        for i in range(1,self.w):
            self.ZL1[i] = self.sigmoid(np.dot(self.WL1[:,i],self.x))
        #use previous z values for second layer.
        for i in range(1,self.w):
            self.ZL2[i] = self.sigmoid(np.dot(self.WL2[:,i], self.ZL1))
        self.y = np.dot(self.WL3, self.ZL2)

    '''
    BackProp will call forwardProp on the given dataset
    '''
    def backProp(self, ys):
        self.forwardProp()

        l3_grad = self.compute_grad(3,self.WL3,ys)
        l2_grad = self.compute_grad(2,self.WL2,ys)
        l1_grad = self.compute_grad(1,self.WL1,ys)
        return l1_grad,l2_grad,l3_grad

    '''
    following instructions from backpropagation slide 63
    '''
    def compute_grad(self,layer_i, layer,ys):
        gradient = np.zeros(layer.shape)
        dl_dy = self.y - ys
        if(layer_i == 1):
            for i in range(0,self.numInputs):
                for j in range(1,self.w):
                    for k in range(1,self.w):
                        z1 = self.ZL1[j]
                        z2 = self.ZL2[k]
                        gradient[i,j] += dl_dy*self.WL3[k]*(z2*(1-z2))*self.WL2[j,k]*(z1*(1-z1))*self.x[i]
            return gradient
        if(layer_i == 2):
            for i in range(0,self.w):
                for j in range(1,self.w):
                    z = self.ZL2[j]
                    gradient[i,j] = dl_dy*self.WL3[j]*(z*(1-z))*self.ZL1[i]
            return gradient
        if(layer_i == 3):
            i = 0
            for val in self.ZL2:
                gradient[i] = val*dl_dy
                i+=1
            return gradient

    def randomize_w(self):
        self.WL1 = np.random.rand(self.numInputs,self.w)
        self.WL2 = np.random.rand(self.w,self.w)
        self.WL3 = np.random.rand(self.w)

    '''
    lr: learning rate should be a lambda function
    '''
    def stochastic_grad_descent(self, lr, data,t):
        loss = np.empty(t)
        for epoch in range(t):
            np.random.shuffle(data)
            for d in data:
                y_star = d[-1]
                if y_star == 0:
                    y_star = -1
                self.x = np.concatenate(([1],d[:-1]))
                grad1,grad2,grad3 = self.backProp(y_star)
                self.WL3 -= lr(epoch)*grad3
                self.WL2 -= lr(epoch)*grad2
                self.WL1 -= lr(epoch)*grad1
            loss_val = self.calc_loss(data)
            loss[epoch] = loss_val
        return loss

    def calc_loss(self,data):
        sum = 0
        for d in data:
            y_star = d[-1]
            if y_star == 0:
                y_star = -1
            self.input_vals = np.concatenate(([1],d[:-1]))
            self.forwardProp()
            sum += 0.5*(self.y-y_star)**2
        return sum/len(data)

    def calc_error(self, data,test):
        training_error = 0
        test_error = 0
        for d in data:
            self.x = np.concatenate(([1],d[:-1]))
            self.forwardProp()
            if self.y > 0:
                if d[-1] == 0:
                    training_error+=1
            else:
                if d[-1] == 1:
                    training_error+=1
        for d in test:
            self.x = np.concatenate(([1],d[:-1]))
            self.forwardProp()
            if self.y > 0:
                if d[-1] == 0:
                    test_error+=1
            else:
                if d[-1] == 1:
                    test_error+=1
        return training_error/len(data),test_error/len(test)