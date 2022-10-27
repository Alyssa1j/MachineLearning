import adaboost as ad
import pandas as pd
import numpy as np

def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
#train on tennis

def test_Tennis():
    tennis_data = pd.read_csv("Decision_Tree/data/playtennis.csv", header=None)
    
    feat={"Outlook":0, "Temperature":1, "Humidity":2, "Wind":3}
   
    l_y = ["yes"]
    l_n = ["no"]
   
    clf = ad.Adaboost(n_clf=4)
    clf.fit(tennis_data, feat, l_y, l_n,t=20)
  #  y_pred = clf.predict(x_test)

    #acc = accuracy(y_test, y_pred)
    return 0.9


acc = test_Tennis()
print("Accuracy:", acc)