
import numpy as np

import tree_node as tree
import id3_math as m


def ID3(data, attrs, label_yes, label_no, answer, depth=0):
    root = tree.Node()

    max_gain = 0
    max_feat = ""
    for feature in attrs:
        gain = m.info_gain(data, attrs[feature], label_yes, label_no, answer)
        if gain > max_gain:
            max_gain = gain
            max_feat = feature
    root.value = max_feat
    uniq = np.unique(data[attrs[max_feat]])
    for u in uniq:
        subdata = data[data[attrs[max_feat]] == u]

        if m.entropy(subdata, label_yes, answer) == 0.0:
            newNode = tree.Node()
            newNode.isLeaf = True
            newNode.value = u
            newNode.pred = np.unique(subdata[answer])
            root.children.append(newNode)
        else:
            dummyNode = tree.Node()
            dummyNode.value = u
            new_attrs = attrs.copy()
            new_attrs.pop(max_feat)
            child = ID3(subdata, new_attrs, label_yes, label_no, answer)
            dummyNode.children.append(child)
            root.children.append(dummyNode)
    return root


def ID3_ME(data, attrs, label_yes, label_no, answer, depth=0):
    root = tree.Node()

    max_gain = 0
    max_feat = ""
    for feature in attrs:
        gain = m.gain_ME(data, attrs[feature], label_yes, label_no, answer)
        print(feature+":")
        print(gain)
        if gain > max_gain:
            max_gain = gain
            max_feat = feature
    root.value = max_feat
    uniq = np.unique(data[attrs[max_feat]])
    for u in uniq:
        subdata = data[data[attrs[max_feat]] == u]

        if m.ME(subdata, label_yes, answer) == 0.0:
            newNode = tree.Node()
            newNode.isLeaf = True
            newNode.value = u
            newNode.pred = np.unique(subdata[answer])
            root.children.append(newNode)
        else:
            dummyNode = tree.Node()
            dummyNode.value = u
            new_attrs = attrs.copy()
            new_attrs.pop(max_feat)
            child = ID3_ME(subdata, new_attrs, label_yes, label_no, answer)
            dummyNode.children.append(child)
            root.children.append(dummyNode)
    return root

def ID3_GI(data,attrs, label_yes, label_no, answer, depth=0):
    root = tree.Node()

    max_gain = 0
    max_feat = ""
    for feature in attrs:
        gain = m.gain_GI(data, attrs[feature], label_yes, label_no, answer)
        if gain > max_gain:
            max_gain = gain
            max_feat = feature
    root.value = max_feat
    uniq = np.unique(data[attrs[max_feat]])
    for u in uniq:
        subdata = data[data[attrs[max_feat]] == u]

        if m.GI(subdata, label_yes, answer) == 0.0:
            newNode = tree.Node()
            newNode.isLeaf = True
            newNode.value = u
            newNode.pred = np.unique(subdata[answer])
            root.children.append(newNode)
        else:
            dummyNode = tree.Node()
            dummyNode.value = u
            new_attrs = attrs.copy()
            new_attrs.pop(max_feat)
            child = ID3_ME(subdata, new_attrs, label_yes, label_no, answer)
            dummyNode.children.append(child)
            root.children.append(dummyNode)
    return root

def ID3_Method(method, data,attrs, label_yes, label_no, answer, depth=0):
    if method == "ME":
        return ID3_ME(data,attrs, label_yes, label_no, answer, depth)
    if method == "GI":
        return ID3_GI(data,attrs, label_yes, label_no, answer, depth)
    else:
        return ID3(data,attrs, label_yes, label_no, answer, depth)

def printTree(root, depth=0):
        for i in range(depth):
            print("\t", end="")
        print(root.value, end="")
        if root.isLeaf:
            print(" -> ", root.pred)
        print()
        for child in root.children:
            printTree(child, depth + 1)

def prediction_result(root, dataset, features,label_yes,label):
    rows,colms = dataset.shape
    correct =0
    for _,data_row in dataset.iterrows():
        
        b = prediction(root, data_row, features, label_yes,label)
        if b:
            correct+=1
        

    return correct/colms

def prediction(root, data_row, features,label_yes,label):
    #print("data_row:",data_row)
    if root.isLeaf:
        if root.pred == data_row[features[label]]:
            return True
        elif (root.value in label_yes) and (data_row[features[label]] in label_yes):
            return True
        else:
            return False
    else:

        for c in root.children:
            print("Value:" ,c.value)
            print("pred:" ,c.pred)
            print(features[c.value])
            print("Col:", data_row[features[c.value]])
            if c.value == data_row[features[c.value]]:
                return prediction(c.children[0], data_row, label_yes,label)
    return False