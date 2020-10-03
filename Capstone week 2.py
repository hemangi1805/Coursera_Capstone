#!/usr/bin/env python
# coding: utf-8

# Data Understanding

# A description of the data and how it will be used to solve the problem.
# The raw data we will use is provided by the SDOT Traffic Management Division and contains data of all types of collisions that happened in Seattle city from 2004 to May/2020.
# 
# The data contains 194,673 samples and have 37 features that covers the weather and road conditions, collision factors and fatality.
# 
# Let's have a look on the data and understand better how to find the answer to this problem.

# In[33]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

#path = "https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Data-Collisions.csv"
path = "C://Users//Hemangi//Documents//Jupyter//Data-Collisions.csv"
df = pd.read_csv(path)
df


# In[34]:


print('Samples:', df.shape[0])
print('Features:', df.shape[1])


# In[3]:


df.describe(include="all")


# In[35]:


df.isna().sum().to_frame().rename(columns={0: 'NaN Count'})


# In[5]:


df['SEVERITYCODE'].value_counts().to_frame()


# In[36]:


df['SEVERITYDESC'].value_counts().to_frame()


# In[37]:


df['SEVERITYDESC'].value_counts().plot(kind='bar')
plt.xticks(rotation=0)


# Annual amount of traffic incidents in Seattle We notice there is a considerably high amount of incidents only discrepancy is from 2020 as it was recorded incidents that occured till May/2020 not a whole year like the others. We can also infer from the plots that no injury collisions are always more likely to happen.

# In[38]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 6))

df['year'] = pd.DatetimeIndex(df['INCDATE']).year
df['year'].value_counts().sort_index()#.plot(kind='bar')
sns.countplot(x="year", data=df, ax=ax1)
sns.countplot(x="year", hue="SEVERITYDESC", data=df, ax=ax2)
plt.xticks(rotation=45)
ax1.set_title('Annual traffic incidents in Seattle (total)')
ax2.set_title('Annual traffic incidents in Seattle by Severity')


# In[39]:


df['COLLISIONTYPE'].value_counts().sort_values(ascending=False).to_frame()


# In[10]:


sns.countplot(x="COLLISIONTYPE", hue="SEVERITYDESC", data=df)
plt.xticks(rotation=45)
plt.title('Collision Type Occurance')


# Weather condition
# Considering Seattle weather conditions, we notice most incidents happened in a Clear weather. That could be because drivers are less careful when there is no harsh weather condition. It would be interesting to check the correlation between WEATHER and INATTENTIONIND(whether or not collision was due to inattention), but there are too many missing values, 85% of the data is missing.

# In[40]:


df['WEATHER'].value_counts().sort_values(ascending=False).to_frame()


# The condition of the road during the collision
# More occurences in normal road conditions.

# In[41]:


sns.countplot(x="LIGHTCOND", hue="SEVERITYDESC", data=df)
plt.xticks(rotation=90)


# Methodology
# 1. Data preparation and cleaning
# Data cleaning procedure to make the dataset readable and suitable to the machine learning algorithms.
# 
# Dropping all the irrelevant variables and attributes
# Out of the 37 attributes, I will not consider the features with over 40% of missing data, other unclear and irrelevant/noisy variables to our problem. I'll use COLLISIONTYPE, WEATHER, ROADCOND, LIGHTCOND and UNDERINFL as attributes to classify SEVERITYCODE.
# 
# Dealing with missing values
# As my chosen attributes have about 3% of missing data I'll just drop them. I'll still have a considerable amount of data.
# 
# Treating the categorical variables
# In my case, all attributes are categorical. In this step, I will apply label encoding technique for all of them.
# 
# Train/Test split and data normalization
# Now that I treated all my variables I'll separate my independent variables to dataset A and dependent variable 'SEVERITYCODE' to dataset B. After, I'll use this data to randomly pick samples and split in this ratio:
# 
# 70% to train my model
# 30% to test my model Following the split I'll normalize all data to make sure my features are on a similar scale.
# 2. Classification: Modeling and Evaluation
# The prepared dataset will be used to model 3 classification models.
# 
# Logistic Regression: Classifies data by estimating the probability of classes.
# Decision Tree: Classifies by breaking down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed.
# KNN: Classifies unseen data through the majority of its 'neighbours'. In this case we already know K=2 (2 classes of SEVERITY CODES). After obtaining each model's predictions we will evaluate their accuracy, precison, f1-score, log-loss and compare and discuss the results.
# 3. Discussion and Conclusion
# After obtaining the results and evaluating them, in this section I will brief any observations noted based on the results. Finally, will conclude the results of this analysis.

# 1. Data preparation and cleaning
# Dropping all the irrelevant variables and attributes and dealing with missing values.

# In[42]:


data = df[['COLLISIONTYPE', 'WEATHER', 'ROADCOND', 'LIGHTCOND', 'UNDERINFL', 'SEVERITYCODE']]
data = data.dropna()
data.shape


# Convert Categorical features to numerical values

# In[43]:


data['UNDERINFL'].replace(to_replace=['N','Y','0'], value=[0,1,0],inplace=True)
data['UNDERINFL'].value_counts()


# In[44]:


from sklearn.preprocessing import LabelEncoder

features = data[['COLLISIONTYPE', 'WEATHER', 'ROADCOND', 'LIGHTCOND', 'UNDERINFL']]

for feature in ['COLLISIONTYPE', 'WEATHER', 'ROADCOND', 'LIGHTCOND']:
    features[feature] = features[feature].astype('|S') 
    features[feature] = LabelEncoder().fit_transform(features[feature])

features.head()


# In[46]:


X = features
y = data['SEVERITYCODE'].values


# Train/Test split and data normalization

# In[47]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.head()


# In[28]:


from sklearn import preprocessing
X= preprocessing.StandardScaler().fit(X).transform(X)
X_train = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))
X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test.astype(float))
X_train[0:5]
X_test[0:5]


# 2. Classification: Modeling and Evaluation

# KNN

# In[48]:


from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors = 2).fit(X_train, y_train)
model_knn


# Logistic Regression

# In[49]:


from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(C=0.0001, solver='liblinear')
model_lr.fit(X_train, y_train)
model_lr


# Decision Tree

# In[50]:


from sklearn.tree import DecisionTreeClassifier

model_tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
model_tree.fit(X_train, y_train)
model_tree


# Model Evaluation using Test set

# In[ ]:




