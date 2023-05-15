#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


data = pd.read_csv(r"C:\Users\91702\Downloads\data.csv")


# DATA ANALYSING

# In[4]:


data.head()
# data.columns


# CHECKING FOR NULL VALUES

# In[5]:


data.isnull().sum()


# In[6]:


data.drop(["Unnamed: 32","id"],inplace=True,axis = 1)


# In[7]:


data.describe()


# In[8]:


data.groupby(by="diagnosis").mean()
# as we see values for benign are lower than that of malignant.


# SETTING TARGET COLUMN

# In[9]:


target = data["diagnosis"]


# In[10]:


data.drop("diagnosis",axis = 1,inplace=True)


# In[11]:



target = pd.get_dummies(target)
target.drop("B",inplace=True,axis=1)


# In[27]:


target


# CHECKING RELATIONSHIP BETWEEN FACTORS

# In[12]:


from matplotlib import pyplot as plt
import seaborn as sns


# In[13]:


plt.hist(target)


# In[14]:


fig, axes = plt.subplots(nrows=2,ncols=4,figsize=(8,8))

sns.histplot(x=data["radius_mean"],hue=target.values.ravel(),kde=True,ax=axes[0,0])
sns.histplot(x=data["texture_mean"],hue=target.values.ravel(),kde=True,ax=axes[0,1])
sns.histplot(x=data["perimeter_mean"],hue=target.values.ravel(),kde=True,ax=axes[0,2])
sns.histplot(x=data["area_mean"],hue=target.values.ravel(),kde=True,ax=axes[0,3])
sns.histplot(x=data["compactness_mean"],hue=target.values.ravel(),kde=True,ax=axes[1,0])
sns.histplot(x=data["concavity_mean"],hue=target.values.ravel(),kde=True,ax=axes[1,1])

sns.histplot(x=data["concave points_mean"],hue=target.values.ravel(),kde=True,ax=axes[1,3])


# In[15]:


# From above graphs we can see radius_mean,perimeter_mean,area_mean,concavity_mean ,concave points mean affect whether the tumur is b or m


# In[16]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# In[17]:


scaler = StandardScaler()


# In[18]:


scaler.fit(data)
scaled_data = scaler.transform(data)


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


x_train,x_test,y_train,y_test = train_test_split(scaled_data,target,test_size=0.3,random_state=42)


# ## SVM

# In[21]:


from sklearn.svm import SVC


# In[22]:


model = SVC(C=0.5)


# In[23]:


model.fit(x_train,y_train.values.ravel())


# In[24]:


model.score(x_test,y_test)


# In[25]:


y_pred = model.predict(x_test)


# In[26]:


f2 = f1_score(y_test,y_pred)


# In[ ]:


f2


# ## Decision Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[ ]:


model.fit(x_train,y_train.values.ravel())


# In[ ]:


model.score(x_test,y_test)


# ## Random Forest
# using gridsearch for best parameters

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


rcf = RandomForestClassifier(max_depth = 15,max_features=2)


# In[ ]:


forest_params = {"max_depth":[5,10,15,20,25],"max_features":[5,10,15,20,30]}
clf = GridSearchCV(rcf,forest_params,cv=10)


# In[ ]:


clf.fit(x_train,y_train.values.ravel())


# In[ ]:


clf.best_params_


# In[ ]:


rcf.fit(x_train,y_train.values.ravel())


# In[ ]:


y_pred = rcf.predict(x_test)


# In[ ]:


from sklearn.metrics import f1_score


# In[ ]:


f1 = f1_score(y_test,y_pred)


# In[ ]:


f1


# In[ ]:




