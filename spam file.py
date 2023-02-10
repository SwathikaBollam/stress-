#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer


# In[3]:


df=pd.read_csv("C:/Users/swath/OneDrive/Desktop/talent battle/project/week 4/spam.csv",encoding="latin-1")


# In[4]:


df.head(n=10)


# In[5]:


df.shape


# In[6]:


#to check whether target attribute is binary or not
np.unique(df['class'])


# In[8]:


#creating sparse matrix

x=df["message"].values
y=df["class"].values


# In[9]:


#create count vectorizer object
cv=CountVectorizer()

x=cv.fit_transform(x)
v=x.toarray()
print(v)


# In[10]:


first_col=df.pop('message')
df.insert(0,'message',first_col)
df


# In[11]:


#splitting train + test  3:1

train_x=x[:4180]
train_y=y[:4180]

test_x=x[4180:]
test_y=y[4180:]


# In[12]:


bnb=BernoulliNB(binarize=0.0)
model=bnb.fit(train_x,train_y)
y_pred_train=bnb.predict(train_x)
y_pred_test=bnb.predict(test_x)


# In[13]:


print(bnb.score(train_x,train_y)*100)


# In[14]:


print(bnb.score(test_x,test_y)*100)


# In[15]:


from sklearn.metrics import classification_report
print(classification_report(train_y,y_pred_train))


# In[16]:


from sklearn.metrics import classification_report
print(classification_report(test_y,y_pred_test))


# In[ ]:




