import torch
from numpy import load
import pandas as pd
from torch import nn
import numpy as np
import json
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE;
from torch.utils.data import DataLoader
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

embedding = []
label_enc = []
data_count={}
count = 1
with open("output/embedding_extra_clean_seq_60.json",'r') as fr:
    for i in fr:
        i = json.loads(i)
        em, lab = i['embedding'], i['label']
        if lab in data_count:
            if data_count[lab]<10000:
                data_count[lab]+=1
            else:
                continue
        else:
            data_count[lab] = 1
        
        if count%1000 == 0:
            print(data_count)
        count+=1

        embedding.append(i['embedding'])
        label_enc.append(i['label'])

X_sm = embedding
y_sm = label_enc
smote = SMOTE();
X_sm, y_sm = smote.fit_resample(embedding, label_enc)
del embedding
del label_enc
X_train, X_test, y_train, y_test = train_test_split(X_sm,y_sm,test_size=0.2,random_state=42)
m = XGBClassifier(
    max_depth=2,
    gamma=2,
    eta=0.8,
    reg_alpha=0.5,
    reg_lambda=0.5
)
m.fit(X_train, y_train)
y_pred = m.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("F1-score", f1_score(y_test, y_pred, average='macro'))