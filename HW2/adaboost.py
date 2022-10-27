import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'Decision_Tree'))
import numpy as np
import DecisionStump
import id3
import tree_node
# Define AdaBoost class
class Adaboost:
    def __init__(self, n_clf=5): 
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, data, features, pos_label,neg_label, t=100):
        n_samples = len(data)
        _,ly=data.shape

         # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))
        self.clfs = []

        #print(len(data),x,y)
        for i in range(1,3):
            clf = DecisionStump.DecisionStump()
            min_error = float("inf")  
            clf.tree = id3.ID3(data,features,pos_label,neg_label,ly-1,2)
            clf.feature_idx = clf.tree.value
            #calculate error rate for stump:
            for i,data_row in data.iterrows():
                b = id3.prediction(clf.tree, data_row, features, pos_label,ly-1)
                if not b:
                    error = w[i]
                    if(error < min_error):
                        min_error = error
           
            

            clf.alpha = 0.5 * np.log((1.0 - min_error) / (min_error))

            # calculate predictions and update weights


