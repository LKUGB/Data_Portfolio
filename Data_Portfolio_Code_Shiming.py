#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


# In[2]:


data = pd.DataFrame()
data = pd.read_csv('D:/HW and Tests/Fall_2021_curriculums/Data Mining/data/active_players.csv')


# In[3]:


data.head


# In[4]:


data.shape


# In[5]:


data.columns


# In[6]:


data1 = pd.DataFrame()
data1 = data.drop(['Team','Position','College'],1)


# In[7]:


data1


# In[8]:


data1 = data1[data1['Salary']> 0]


# In[9]:


data1


# In[10]:


data1.describe()


# In[11]:


data1.columns


# In[12]:


data1.describe


# In[13]:


data1Normalized = data1.copy()
  
# normalization using the max value
for column in data1Normalized[['Age','Height','Weight','Salary']]:
    data1Normalized[column] = data1Normalized[column]  / data1Normalized[column].max()
      
# view normalized data
display(data1Normalized)


# In[14]:


def euclidean_dist(v1, v2):
    dist = 0
    for i in range(len(v1)):
        dist += pow(v1[i] - v2[i],2)
    return dist


# In[15]:


def inner_product(v1, v2):
    product = 0
    for i in range(len(v1)):
        product += v1[i] * v2[i]
    return product


# In[16]:


def vect_len(v1):
    temp = 0.0
    for i in range(len(v1)):
        temp += v1[i]*v1[i]
    return(math.sqrt(temp))


# In[17]:


def cosine_sim(v1, v2):
    lenV1 = vect_len(v1)
    lenV2 = vect_len(v2)
    innerV1V2 = inner_product(v1, v2)
    cosV1V2 = innerV1V2/(lenV1 * lenV2)
    return cosV1V2


# In[18]:


dataName = pd.DataFrame()
dataName = data1Normalized.loc[:,'Name']
display(dataName)


# In[19]:


dataNameArr = dataName.to_numpy()
dataNameArr


# In[20]:


data2 = pd.DataFrame()
data2 = data1Normalized.drop(['Name'],1)
display(data2)


# In[21]:


data2Arr = data2.to_numpy()
data2Arr


# In[22]:


# define a similarity matrix that has n x 4 columns (col1 and col2 are the two players being compared; col3 and col4 are
# the similarity values derived from the two similarity measures)
playerSim = []
maxDist = euclidean_dist(data2Arr[0], data2Arr[1])
for i in range(len(data2Arr)):
    for j in range(i+1, len(data2Arr)):
        dist = euclidean_dist(data2Arr[i], data2Arr[j])
        if dist > maxDist:
            maxDist = dist
        cosSim = cosine_sim(data2Arr[i], data2Arr[j])
        playerSim.append([dataNameArr[i], dataNameArr[j], dist, cosSim])

# Scale the Euclidean distance to [0, 1]
# convert to similarity value.
for i in range(len(playerSim)):
    playerSim[i][2] = 1.0 - playerSim[i][2]/(maxDist)


# In[23]:


playerSim


# In[24]:


EuclideanCosArr = pd.DataFrame(playerSim, columns = ['player_1','player_2', 'euclidean_distance', 'cosine_similarity'])


# In[25]:


EuclideanCosArr.to_csv('D:/HW and Tests/Fall_2021_curriculums/Data Mining/data/player_sim.csv')


# In[29]:


plt.scatter(EuclideanCosArr["euclidean_distance"], EuclideanCosArr["cosine_similarity"], alpha = 0.6)
plt.title("Euclidean Distance VS Cosine Similarity")
plt.xlabel("Euclidean")
plt.ylabel("Cosine")
plt.show()


# In[ ]:




