#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv('data_science_job.csv')


# In[4]:


df.head()


# In[7]:


df.isnull().mean()*100 #finding percentage of missing data in each col


# In[9]:


#we will use CCA on city, enrolled, educ,experience, training hours
df.shape


# In[10]:


cols= [var for var in df.columns if df[var].isnull().mean()<0.05 and df[var].isnull().mean()>0]
cols


# In[12]:


df[cols].sample(5)


# In[18]:


colss= ['enrolled_university', 'education_level']
for col in colss:
    print(df[col].value_counts())
    print()


# In[20]:


len(df[cols].dropna())/len(df) #after removing these values, we're left with 89% of data


# In[21]:


new_df= df[cols].dropna()
df.shape, new_df.shape


# In[23]:


new_df.hist(bins=50, density=True, figsize=(12,12)) #plot for numerical data
plt.show()


# In[26]:


fig= plt.figure()
ax= fig.add_subplot(111)

#original data
df['training_hours'].hist(bins=50, ax=ax, color='green', density=True, alpha=0.8)

#data after cca, the argument alpha makes the color transparent, so we can see the overlay of the 2 distn
new_df['training_hours'].hist(bins=50, ax=ax, color='red', density=True, alpha=0.8)

#little portion of green is visible that is cca is performed well


# In[27]:


fig= plt.figure()
ax= fig.add_subplot(111)

#original data
df['city_development_index'].hist(bins=50, ax=ax, color='green', density=True, alpha=0.8)

#data after cca, the argument alpha makes the color transparent, so we can see the overlay of the 2 distn
new_df['city_development_index'].hist(bins=50, ax=ax, color='red', density=True, alpha=0.8)

#little portion of green is visible that is cca is performed well


# In[28]:


fig= plt.figure()
ax= fig.add_subplot(111)

#original data
df['experience'].hist(bins=50, ax=ax, color='green', density=True, alpha=0.8)

#data after cca, the argument alpha makes the color transparent, so we can see the overlay of the 2 distn
new_df['experience'].hist(bins=50, ax=ax, color='red', density=True, alpha=0.8)

#little portion of green is visible that is cca is performed well


# In[29]:


temp= pd.concat([
    #percent of obsv per category, original data
    df['enrolled_university'].value_counts()/len(df),
    
    #percent of obsv per categ, new data
    new_df['enrolled_university'].value_counts()/len(new_df)
], axis=1)

#add col names
temp.columns=['original', 'cca']
temp


# In[30]:


temp= pd.concat([
    #percent of obsv per category, original data
    df['education_level'].value_counts()/len(df),
    
    #percent of obsv per categ, new data
    new_df['education_level'].value_counts()/len(new_df)
], axis=1)

#add col names
temp.columns=['original', 'cca']
temp


# In[ ]:




