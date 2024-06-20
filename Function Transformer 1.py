#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[80]:


import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer


# In[81]:


df= pd.read_csv('train (1).csv', usecols=['Age','Fare','Survived'])


# In[82]:


df.head()


# # analyze missing values

# In[83]:


df.isnull().sum()


# In[84]:


df['Age'].fillna(df['Age'].mean(), inplace=True)


# In[85]:


df.isnull().sum()


# # Define x and y

# In[86]:


x= df.iloc[:, 1:3]
y=df.iloc[:, 0]


# In[87]:


x.head()


# In[88]:


y.head()


# In[89]:


x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=45)


# # Plot the current age distn and QQ plot

# In[90]:


plt.figure(figsize=(12,5))
plt.subplot(121)
sns.distplot(x_train['Age'])
plt.title('x_train age pdf')

plt.subplot(122)
stats.probplot(x_train['Age'], dist='norm', plot=plt)
plt.title('Age qq plot')

plt.show()  #distn is almost normal naturally


# # Plot current fare distn and QQ plot

# In[91]:


plt.figure(figsize=(12,4))
plt.subplot(121)
sns.distplot(x_train['Fare'])
plt.title('Fare pdf')

plt.subplot(122)
stats.probplot(x_train['Fare'], dist='norm', plot=plt)
plt.title('Fare QQ plot')

plt.show() #not normal naturally


# # Define classifiers

# In[92]:


clf1= LogisticRegression()
clf2= DecisionTreeClassifier()
clf3= LinearRegression()


# In[93]:


#without transforming it, let's check its accuracy score by both algos
clf1.fit(x_train, y_train)
clf2.fit(x_train, y_train)
clf3.fit(x_train, y_train)

y_pred1= clf1.predict(x_test)
y_pred2= clf2.predict(x_test)
y_pred3= clf2.predict(x_test)

print('Accuracy LoR', accuracy_score(y_test, y_pred1))
print('Accuracy DT', accuracy_score(y_test, y_pred2))
print('Accuracy LR', accuracy_score(y_test, y_pred3))
#naturally the accuracy of age is higher


# # Use log transform and then check accuracy for both classifier

# In[94]:


#since fare is rightly skewed, use log transformation first
trf= FunctionTransformer(func= np.log1p)


# In[95]:


x_train_transf= trf.fit_transform(x_train)
x_test_transf= trf.transform(x_test)


# In[96]:


clf1.fit(x_train_transf, y_train)
clf2.fit(x_train_transf, y_train)

y_pred1= clf1.predict(x_test_transf)
y_pred2= clf2.predict(x_test_transf)

print('accuracy LR', accuracy_score(y_test, y_pred1))
print('accuracy DT', accuracy_score(y_test, y_pred2))


# In[97]:


#again do the same using cross validation
x_transf= trf.fit_transform(x)
print('LoR', np.mean(cross_val_score(clf1, x_transf, y, scoring='accuracy', cv=20)))
print('DT', np.mean(cross_val_score(clf2, x_transf, y, scoring='accuracy', cv=20)))


# In[98]:


plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
stats.probplot(x_train['Fare'], dist= 'norm', plot= plt)
plt.title('fare before log')

plt.subplot(122)
stats.probplot(x_train_transf['Fare'], dist='norm', plot=plt)
plt.title('fare after log')

plt.show()

               


# In[99]:


plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
stats.probplot(x_train['Age'], dist= 'norm', plot= plt)
plt.title('age before log')

plt.subplot(122)
stats.probplot(x_train_transf['Age'], dist='norm', plot=plt)
plt.title('Age after log')

plt.show()


# In[100]:


# use other transformation
def apply_transform(transform):
    x= df.iloc[:, 1:3]
    y= df.iloc[:, 0]
    trf= ColumnTransformer([('log', FunctionTransformer(transform), ['Fare'])], remainder='passthrough')
    x_transf= trf.fit_transform(x)
    
    print('accuracy', np.mean(cross_val_score(clf1, x_transf, y, scoring='accuracy', cv=10)))
    
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    stats.probplot(x['Fare'], dist= 'norm', plot= plt)
    plt.title('fare before transf')
    
    plt.subplot(122)
    stats.probplot(x_transf[:,0], dist='norm', plot=plt)
    plt.title('fare after transf')
    
    plt.show()
        


# In[104]:


apply_transform(lambda x: 1/(x+0.00001))

