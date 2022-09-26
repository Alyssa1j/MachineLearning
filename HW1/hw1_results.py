import pandas as pd
import id3


#data columns are
#0      1       2       3       4           5       6
#buying, maint, doors, persons, lug_boot, safety, label
buying= ["low", "med", "high", "vhigh"]
maint = ["low", "med", "high", "vhigh"]
doors = ["2", "3", "4", "5more"]
person = ["2", "4", "more"]
lug_boot = ["small", "med", "big"]
safety= ["low", "med", "high"]

features = {"buying":0, "maint":1, "doors":2, "person":3, "lug_boot":4, "safety":5}
label_yes = ["acc", "good", "vgood"]
label_no = ["unacc"]


def car_train(method, depth=6):
    #changing the label to be binary 
    car_data = pd.read_csv("data/train.csv", header=None)
    root = id3.ID3_Method(method,car_data, features, label_yes, label_no, 6,6)
    return root

def car_test(method, depth=6):
    test_data = pd.read_csv("data/test.csv", header=None)
    test_root = id3.ID3_Method(method,test_data, features, label_yes, label_no, 6,6)
    return test_root

def test_Tennis(method,depth=6):
    tennis_data = pd.read_csv("data/PlayTennis.csv", header=None)
    
    feat={"Outlook":0, "Temperature":1, "Humidity":2, "Wind":3}
   
    l_y = ["yes"]
    l_n = ["no"]
    tennis_root = id3.ID3_Method(method,tennis_data, feat, l_y, l_n, 4,6)
    return tennis_root

root_train_IG = car_train("IG")
root_train_ME = car_train("ME")
root_train_GI = car_train("GI")
root_tennis = test_Tennis("IG")
id3.printTree(root_train_IG)
id3.printTree(root_train_ME)
id3.printTree(root_train_GI)
id3.printTree(root_tennis)
