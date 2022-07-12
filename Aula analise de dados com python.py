#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[7]:


path = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"


# In[8]:


df= pd.read_csv(path, header=None)


# In[25]:


df


# In[10]:


df.head(10)


# In[12]:


df.tail(5)


# In[13]:


headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels",  "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]


# In[14]:


df.columns= headers


# In[17]:


df.head(5)


# In[27]:


df.to_csv('auto.csv')


# In[22]:


path = "C:/Users/Ana Januário/Desktop" #não consegui salvar o csv na pasta que eu queria


# In[23]:


df.to_csv(path) #não consegui salvar o csv na pasta que eu queria


# In[28]:


df.dtypes #meus tipos de dados


# In[29]:


df.describe() #descritiva  - equivale ao basicstats no Rstudio


# In[30]:


df.describe(include="all") #força descritiva de todas as colunas, mesmo as não numericas


# In[31]:


df.info() #outra forma de ver informações sobre os dados


# In[ ]:




