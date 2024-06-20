#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


# In[23]:


df= pd.read_csv('train (1)_toy.csv', usecols=['Age','Fare','Survived','SibSp','Parch'])


# In[24]:


df['family']= df['SibSp']+df['Parch']


# In[25]:


df=df.drop(columns=['SibSp','Parch'])


# In[26]:


df.head()


# In[27]:


df.isnull().sum()


# In[28]:


df.isnull().mean()*100


# In[29]:


x= df.iloc[:,1:]
y=df.iloc[:,0]


# In[30]:


x


# In[31]:


y


# In[34]:


x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)


# In[36]:


x_train.shape, x_test.shape


# In[37]:


x_train.isnull().mean()


# In[38]:


mean_age= x_train['Age'].mean()
median_age=x_train['Age'].median()

mean_fare= x_train['Fare'].mean()
median_fare= x_test['Fare'].median()


# In[39]:


x_train['Age_mean']= x_train['Age'].fillna(mean_age)
x_train['Age_median']= x_train['Age'].fillna(median_age)

x_train['Fare_mean']= x_train['Fare'].fillna(mean_fare)
x_train['Fare_median']= x_train['Fare'].fillna(median_fare)


# In[43]:


x_train.sample(5)


# In[46]:


print('original age variable variance:', x_train['Age'].var())
print('after mean imputation age variable variance:', x_train['Age_mean'].var())
print('after median age variable variance:', x_train['Age_median'].var())
print()
print('original fare variable variance:', x_train['Fare'].var())
print('after mean imputation fare variable variance:', x_train['Fare_mean'].var())
print('after median fare variable variance:', x_train['Fare_median'].var())


# In[50]:


fig= plt.figure()
ax= fig.add_subplot(111)

#original variance distn
x_train['Fare'].plot(kind='kde', ax=ax)

#variable imputed with the mean
x_train['Fare_mean'].plot(kind='kde', color='red', ax=ax)

#variable imputed with the median
x_train['Fare_median'].plot(kind='kde', color='green', ax=ax)

#add legends
lines, labels= ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')


# In[51]:


fig= plt.figure()
ax= fig.add_subplot(111)

#original variance distn
x_train['Age'].plot(kind='kde', ax=ax)

#variable imputed with the mean
x_train['Age_mean'].plot(kind='kde', color='red', ax=ax)

#variable imputed with the median
x_train['Age_median'].plot(kind='kde', color='green', ax=ax)

#add legends
lines, labels= ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')


# In[52]:


x_train.cov()


# In[56]:


x_train[['Age','Age_mean','Age_median']].boxplot()


# In[60]:


x_train[['Fare','Fare_mean','Fare_median']].boxplot(grid=False)


# # Using sklearn, imputing

# In[61]:


x_train,x_test,y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=42)


# In[62]:


imputer1= SimpleImputer(strategy='mean')
imputer2= SimpleImputer(strategy='median')


# In[63]:


trf= ColumnTransformer([
    ('imputer1', imputer1,['Age']),
    ('imputer2', imputer2, ['Fare'])
], remainder='passthrough')


# In[64]:


trf.fit(x_train)


# In[65]:


trf.named_transformers_['imputer1'].statistics_


# In[66]:


trf.named_transformers_['imputer2'].statistics_


# In[67]:


x_train= trf.transform(x_train)
x_test= trf.transform(x_test)


# In[68]:


x_train


# # Arbitrary imputation

# In[ ]:


df1= pd.read_csv('train (1)_toy.csv', usecols=['Age','Fare','Survived','SibSp','Parch'])
df1['family']= df['SibSp']+df['Parch']
df1=df.drop(columns=['SibSp','Parch'])


# In[ ]:





# In[ ]:


x_train['Age_99']= x_train['Age'].fillna(99)
x_train['Age_minus1']= x_train['Age'].fillna(-1)

x_train['Fare_999']= x_train['Fare'].fillna(999)
x_train['Fare_minus1']= x_train['Fare'].fillna(-1)


# In[ ]:




