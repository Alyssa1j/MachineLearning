import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'Decision_Tree'))
import numpy as np
import DecisionStump
import id3
import tree_node
# Define AdaBoost class
class Adaboost:
    def __init__(self): 
        self.clfs = []

    def fit(self, data, features, pos_label,neg_label, t=100):
        n_samples = len(data)
        _,ly=data.shape

         # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))
        self.clfs = []

        #print(len(data),x,y)
        for i in range(1,t):
            clf = DecisionStump.DecisionStump()
            t_error = 0
            clf.tree = id3.ID3_W_BT(data,features,pos_label,neg_label,ly-1,w,2)
            clf.feature_idx = clf.tree.value
            #find Et from stump
            for i,data_row in data.iterrows():
                b = id3.prediction(clf.tree, data_row, features, pos_label,ly-1)
                if not b:
                    t_error += w[i]
                    
           # print(t_error)
            
            clf.alpha = 0.5 * np.log10((1.0 - t_error) / (t_error))
          #  print(clf.alpha)
            #increase weight on incorrect predictions
            #decrease weight on correct predictions
            predictions = clf.predict(data,features, pos_label,ly-1)
            w *= np.exp(clf.alpha * predictions)
            # Normalize to one
            w /= np.sum(w)
                 # print("Normalized:",np.unique(w))
            # Save classifier
            self.clfs.append(clf)
           # print(clf.feature_idx,clf.alpha)
    def predict(self, data,feat,ly):
        _,y=data.shape
        clf_preds = [-clf.alpha * clf.predict(data,feat,ly,y-1) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred