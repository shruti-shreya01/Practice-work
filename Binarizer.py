#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer


# In[3]:


df= pd.read_csv('train (1).csv', usecols=['Age', 'Fare', 'SibSp', 'Parch', 'Survived'])


# In[4]:


df.isnull().sum()


# In[5]:


df.dropna(inplace=True)


# In[6]:


df.head()


# In[7]:


df['family']= df['SibSp']+df['Parch']


# In[8]:


df.head()


# In[9]:


df.drop(columns=['SibSp', 'Parch'], inplace=True)


# In[10]:


df.head()


# In[11]:


x=df.drop(columns=['Survived'])
y=df['Survived']


# In[15]:


x_train, x_test,y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=35)
                                         


# In[32]:


x_train.sample(5)


# In[20]:


#without binarization
clf= DecisionTreeClassifier()
clf.fit(x_train,y_train)
y_pred= clf.predict(x_test)
accuracy_score(y_test,y_pred)


# In[21]:


#applying binarization
from sklearn.preprocessing import Binarizer


# In[27]:


trf= ColumnTransformer([('bin', Binarizer(copy=False), ['family'])], remainder='passthrough')


# In[28]:


x_train_trf= trf.fit_transform(x_train)
x_test_trf= trf.transform(x_test)


# In[29]:


pd.DataFrame(x_train_trf, columns=['family','Age','Fare'])


# In[31]:


clf= DecisionTreeClassifier()
clf.fit(x_train_trf,y_train)
y_pred2= clf.predict(x_test)
accuracy_score(y_test,y_pred2)


# In[ ]:




