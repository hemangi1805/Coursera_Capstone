#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

#path = "https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Data-Collisions.csv"
path = "C://Users//Hemangi//Documents//Jupyter//Data-Collisions.csv"
df = pd.read_csv(path)
df


# In[5]:


print('Samples:', df.shape[0])
print('Features:', df.shape[1])


# In[6]:


df.describe(include="all")


# In[7]:


df.isna().sum().to_frame().rename(columns={0: 'NaN Count'})


# What is our target variable?
# Our target variable SEVERITYCODE that corresponds to the severity of the collision:
# 1 Property Damage only collision which is the same as Not injured coliision
# 2 Injury collision By looking to the target variable I know it's a binary classification problem.

# In[8]:


df['SEVERITYCODE'].value_counts().to_frame()


# In[9]:


df['SEVERITYDESC'].value_counts().to_frame()


# In[10]:


df['SEVERITYDESC'].value_counts().plot(kind='bar')
plt.xticks(rotation=0)


# Annual amount of traffic incidents in Seattle
# We notice there is a considerably high amount of incidents only discrepancy is from 2020 as it was recorded incidents that occured till May/2020 not a whole year like the others. We can also infer from the plots that no injury collisions are always more likely to happen.

# In[11]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 6))

df['year'] = pd.DatetimeIndex(df['INCDATE']).year
df['year'].value_counts().sort_index()#.plot(kind='bar')
sns.countplot(x="year", data=df, ax=ax1)
sns.countplot(x="year", hue="SEVERITYDESC", data=df, ax=ax2)
plt.xticks(rotation=45)
ax1.set_title('Annual traffic incidents in Seattle (total)')
ax2.set_title('Annual traffic incidents in Seattle by Severity')


# Collision types
# There is a considerable difference on the collision occurences according to collision types. Being the most recurrent accidents with parked cars,angles and rear ended.

# In[12]:


df['COLLISIONTYPE'].value_counts().sort_values(ascending=False).to_frame()


# In[13]:


sns.countplot(x="COLLISIONTYPE", hue="SEVERITYDESC", data=df)
plt.xticks(rotation=45)
plt.title('Collision Type Occurance')


# Considering Seattle weather conditions, we notice most incidents happened in a Clear weather. That could be because drivers are less careful when there is no harsh weather condition. It would be interesting to check the correlation between WEATHER and INATTENTIONIND(whether or not collision was due to inattention), but there are too many missing values, 85% of the data is missing.

# In[14]:


df['WEATHER'].value_counts().sort_values(ascending=False).to_frame()


# In[15]:


sns.countplot(x="WEATHER", hue="SEVERITYDESC", data=df)
plt.xticks(rotation=45)


# Driver under influence of drugs or alcohol
# In most incidents drivers were not under any influence.

# In[16]:


sns.countplot(x="UNDERINFL", hue="SEVERITYDESC", data=df)
plt.xticks(rotation=45)


# Each feature have a different weight of influence on the severity of the collision. Overall, all of them are consistently infering that no-injury accidents in normal driving conditions are more recurrent.
# 
# We will use COLLISIONTYPE, WEATHER, ROADCOND, LIGHTCOND and UNDERINFL as attributes to classify SEVERITYCODE. For that we will need to prepare this features so it is suitable for a binary classification model. I'll use some popular machine learning algorithms (SVM, Logistic Regression, Naive Bayes and KNN) build up models to analyze their performance and predict the collision severity.

# In[ ]:




