import torch
from numpy import load
import pandas as pd
from torch import nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE;
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
import argparse

def read(path):
    with open(path,'r') as fr:
        embedding = fr.readlines()
        embedding = list(map(eval, embedding))
    return embedding


def label(path):
    df = pd.read_csv(path,chunksize=1000)
    labels = []
    for chunk in df:
        labels = labels + chunk['type'].tolist()
    label_enc = []
    for label in labels:
        label_enc.append(label_encoding[label])
    return label_enc


class MLInfer():
    def __init__(self,X, Y):
        self.X = X
        self.Y = Y

    def label_mask(self,l1,l2):
        Y_mask = []
        c_1 = 0
        c_0 = 0
        for i in self.Y:
            if i in l1:
                Y_mask.append(1)
                c_1+=1
            else:
                Y_mask.append(2)
                c_0+=1
        return Y_mask
    
    def fix_imbalance(self,label):
        print("Masking Label")
        Y_mask = self.label_mask(label[0],label[1])
        print("Fixing imbalance\n")
        smote = SMOTE();
        self.X_sm, self.y_sm = smote.fit_resample(self.X, Y_mask)
        return self.X_sm, self.y_sm
    
    def fit(self,functions,label):
        self.fix_imbalance(label)
        X_train_mask, X_test_mask, Y_train_mask, Y_test_mask = train_test_split(self.X_sm,self.y_sm,test_size=0.2,random_state=42)
        self.X_train_mask = torch.FloatTensor(X_train_mask)
        self.X_test_mask = torch.FloatTensor(X_test_mask)
        self.Y_train_mask = torch.LongTensor(Y_train_mask)
        self.Y_test_mask = torch.LongTensor(Y_test_mask)
        
        if 'xgboost' in functions:
            self.xgboost()
        if 'lr' in functions:
            self.lr()
        if 'svm' in functions:
            self.svm()
    
    
    def xgboost(self):
        print("Training XGBOOST")
        m = XGBClassifier(max_depth=2,gamma=2,eta=0.8,reg_alpha=0.5,reg_lambda=0.5)
        m.fit(self.X_train_mask, self.Y_train_mask)
        print("TESTING XGBOOST")
        y_pred = m.predict(self.X_test_mask)
        accuracy = accuracy_score(self.Y_test_mask, y_pred)
        print("Accuracy: %.2f%% \n" % (accuracy * 100.0))
        print("F1-score", f1_score(self.Y_test_mask, y_pred, average='macro'))
    
    def lr(self):
        print("Training Logistic Regression")
        clf  = LogisticRegression(random_state=42,max_iter=10000,penalty='l2',C=0.1)
        print("Testing Logistic Regression")
        clf.fit(self.X_train_mask, self.Y_train_mask)
        accuracy = clf.score(self.X_test_mask, self.Y_test_mask)
        print("Accuracy: %.2f%% \n" % (accuracy * 100.0))
        print("F1-score", f1_score(self.Y_test_mask, clf.predict(self.X_test_mask), average='macro'))
        
    def svm(self):
        print("Training SVM")
        clf = LinearSVC(random_state=0, tol=1e-5)
        print("Testing SVM")
        clf.fit(self.X_train_mask, self.Y_train_mask)
        accuracy = clf.score(self.X_test_mask, self.Y_test_mask)
        print("Accuracy: %.2f%% \n" % (accuracy * 100.0))
        print("F1-score", f1_score(self.Y_test_mask, clf.predict(self.X_test_mask), average='macro'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remb",default="output/embedding_infersent.txt")
    parser.add_argument("--label",default="/home/amoghmishra23/appledore/MBTI-Personality-Detection/data/mbti_comments/mbti_comments_cleaned.csv")
    args = parser.parse_args()
    
    label_encoding = { "istj":1, "istp":2, "isfj":3, "isfp":4, "infj":5, "infp":6, "intj":7, "intp":8, "estp":9, "estj":10, "esfp":11, "esfj":12, "enfp":13, "enfj":14, "entp":15, "entj":16 }
    e = [9,10,11,12,13,14,15,16]
    i = [1,2,3,4,5,6,7,8]

    s = [1,2,3,4,9,10,11,12]
    n = [5,6,7,8,13,14,15,16]


    j = [1,3,5,7,10,12,14,16]
    p = [2,4,6,8,9,11,13,15]

    t = [1,2,7,8,9,10,15,16]
    f = [3,4,5,6,11,12,13,14]

    X = read(args.remb)
    Y = label(args.label)
    mli = MLInfer(X,Y)
    mli.fit(['xgboost','lr','svm'],[e,i])
    mli.fit(['xgboost','lr','svm'],[s,n])
    mli.fit(['xgboost','lr','svm'],[j,p])
    mli.fit(['xgboost','lr','svm'],[t,f])
    

# ### MACRO-F1 score combining INFERSENT with ML models:
# 
# ## E/I
#     * XG-BOOST: 81.93%
#     * SVM: 78.7%
#     * LR: 65.2%
# 
# ## S/N
#     * XG-BOOST: 88.15%
#     * SVM: 86%
#     * LR: 69.81%
# 
# ## T/F
#     * XG-BOOST: 70.51
#     * SVM: 71.31
#     * LR: 65.2
# 
# ## J/P
#     * XG-BOOST: 64.2%
#     * SVM: 63%
#     * LR: 62.1%