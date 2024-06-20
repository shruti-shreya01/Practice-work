#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


# In[2]:


from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.preprocessing import PowerTransformer


# In[3]:


df= pd.read_csv('concrete_data.csv')


# In[4]:


df.head()


# In[6]:


df.isnull().sum()


# In[8]:


df.describe() #chcek if any value is negative or 0 cz in such case box cos transf will not work


# In[9]:


x= df.drop(columns=['Strength'])
y= df.iloc[:,-1]


# In[11]:


x


# In[12]:


y


# In[18]:


x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=72)


# In[19]:


#applying regression without any transformation
LR= LinearRegression()
LR.fit(x_train, y_train)
y_pred= LR.predict(x_test)
r2_score(y_test, y_pred)


# In[20]:


np.mean(cross_val_score(LR,x,y,scoring='r2'))


# In[21]:


#plotting the distplot without any transformation
for col in x_train.columns:
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.distplot(x_train[col])
    plt.title(col)
    
    plt.subplot(1,2,2)
    stats.probplot(x_train[col], dist='norm', plot=plt)
    plt.title(col)
    plt.show()


# In[25]:


#applying box-cox transformation
pt= PowerTransformer(method='box-cox')
x_train_transf= pt.fit_transform(x_train+0.0001)
x_test_transf= pt.transform(x_test+0.00001)

pd.DataFrame({'cols': x_train.columns, 'box_cox_lambdas': pt.lambdas_})


# In[26]:


#applying linear regression on transformed data
LR= LinearRegression()
LR.fit(x_train_transf, y_train)
y_pred2= LR.predict(x_test_transf)
r2_score(y_test, y_pred2)


# In[27]:


#before and after comparison of box-cox transform
x_train_transf= pd.DataFrame(x_train_transf, columns=x_train.columns)
for col in x_train_transf.columns:
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.distplot(x_train[col])
    plt.title(col)
    
    plt.subplot(122)
    sns.distplot(x_train_transf[col])
    plt.title(col)
    
    plt.show()


# In[29]:


#applying yeo-johnson transform
pt1= PowerTransformer()
x_train_transf2= pt1.fit_transform(x_train)
x_test_transf2= pt1.fit_transform(x_test)

LR= LinearRegression()
LR.fit(x_train_transf2, y_train)
y_pred3= LR.predict(x_test_transf2)
print(r2_score(y_test,y_pred3))

pd.DataFrame({'cols':x_train.columns, 'Yeo_Johnson_lambdas':pt1.lambdas_})




# In[30]:


#applying cross validation score
pt= PowerTransformer()
x_transf2= pt.fit_transform(x)

LR= LinearRegression()
np.mean(cross_val_score(LR, x_transf2, y, scoring='r2'))


# In[34]:


x_train_transf2= pd.DataFrame(x_train_transf2, columns=x_train.columns)


# In[35]:


#before and after comparison of box-cox transform
x_train_transf= pd.DataFrame(x_train_transf2, columns=x_train.columns)
for col in x_train_transf2.columns:
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.distplot(x_train[col])
    plt.title(col)
    
    plt.subplot(122)
    sns.distplot(x_train_transf2[col])
    plt.title(col)
    
    plt.show()


# In[36]:


#side by side both lambdas
pd.DataFrame({'cols':x_train.columns, 'box_cox_lambdas':pt.lambdas_, 'Yeo_Johnson_lambdas':pt1.lambdas_})


# In[ ]:




