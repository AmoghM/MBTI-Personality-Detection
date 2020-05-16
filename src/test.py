#!/usr/bin/env python
# coding: utf-8

# In[16]:


import json
import numpy as np
from sklearn.model_selection import train_test_split


# In[11]:


X_sm = np.load("output/embedding_160k_balanced.npy")


# In[13]:


y_sm = np.load("output/label_160k_balanced.npy")


# In[19]:


from sklearn.model_selection import StratifiedKFold


# In[20]:


skf = StratifiedKFold(n_splits=5,shuffle=True)
skf.get_n_splits(X_sm, y_sm)


# In[22]:


for train_index, test_index in skf.split(X_sm, y_sm):
    X_train, X_test = np.array(X_sm)[train_index], np.array(X_sm)[test_index]
    y_train, y_test = np.array(y_sm)[train_index], np.array(y_sm)[test_index]


# In[ ]:


from xgboost import XGBClassifier

m = XGBClassifier(
    max_depth=2,
    gamma=2,
    eta=0.8,
    reg_alpha=0.5,
    reg_lambda=0.5
)
m.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import accuracy_score
y_pred = m.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


from sklearn.metrics import f1_score
f1_score(y_Test, y_pred, average='macro')

