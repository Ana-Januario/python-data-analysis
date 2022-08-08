#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df = pd.read_csv(path)
df.head()


# In[4]:


#Analyzing individual feature patterns using visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:



# list the data types for each column
print(df.dtypes)


# In[6]:


df.corr()


# In[10]:


a=df[['bore','stroke','compression-ratio','horsepower']]


# In[11]:


a.corr()


# In[13]:


df[['bore','stroke','compression-ratio','horsepower']].corr()


# In[14]:


#Positive Linear Relationship

#Let's find the scatterplot of "engine-size" and "price".

# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)


# In[15]:


df[["engine-size", "price"]].corr()


# In[16]:


sns.regplot(x="highway-mpg", y="price", data=df)


# In[17]:


#As highway-mpg goes up, the price goes down: this indicates an inverse/negative relationship between these two variables. 
#Highway mpg could potentially be a predictor of price.
#We can examine the correlation between 'highway-mpg' and 'price' and see it's approximately -0.704.

df[['highway-mpg', 'price']].corr()


# In[18]:


# Weak Linear Relationship

sns.regplot(x="peak-rpm", y="price", data=df)


# In[19]:


df[['peak-rpm','price']].corr()


# In[21]:


df[["stroke","price"]].corr()


# In[22]:


sns.regplot(x="stroke", y="price", data=df)


# In[23]:


#There is a weak correlation between the variable 'stroke' and 'price.
#' as such regression will not work well. We can see this using "regplot" to demonstrate this.


#Categorical Variables

sns.boxplot(x="body-style", y="price", data=df)


# In[24]:


sns.boxplot(x="engine-location", y="price", data=df)


# In[25]:


# drive-wheels
sns.boxplot(x="drive-wheels", y="price", data=df)


# In[26]:


#Descriptive Statistical Analysis
#Let's first take a look at the variables by utilizing a description method.

#The describe function automatically computes basic statistics for all continuous variables. 
#Any NaN values are automatically skipped in these statistics.

#This will show:

#the count of that variable
#the mean
#the standard deviation (std)
#the minimum value
#the IQR (Interquartile Range: 25%, 50% and 75%)
#the maximum value

#We can apply the method "describe" as follows:

df.describe(include=['object'])


# In[27]:


df['drive-wheels'].value_counts()


# In[28]:


df['drive-wheels'].value_counts().to_frame()


# In[29]:


drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts


# In[30]:


drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts


# In[31]:


# engine-location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)


# In[32]:


df['drive-wheels'].unique()


# In[33]:


df_group_one = df[['drive-wheels','body-style','price']]


# In[34]:


# grouping results
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
df_group_one


# In[35]:


#You can also group by multiple variables. For example, let's group by both 'drive-wheels' and 'body-style'. 
#This groups the dataframe by the unique combination of 'drive-wheels' and 'body-style'. 
#We can store the results in the variable 'grouped_test1'.


# grouping results
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_test1


# In[36]:


#This grouped data is much easier to visualize when it is made into a pivot table. 
#A pivot table is like an Excel spreadsheet, with one variable along the column and another along the row. 
#We can convert the dataframe to a pivot table using the method "pivot" to create a pivot table from the groups.

grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot


# In[37]:


grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
grouped_pivot


# In[45]:


#Use the "groupby" function to find the average "price" of each car based on "body-style".

# grouping results
df_gptest2 = df[['body-style','price']]
grouped_test_bodystyle = df_gptest2.groupby(['body-style'],as_index= False).mean()
grouped_test_bodystyle


# In[46]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[47]:


#Variables: Drive Wheels and Body Style vs. Price

#Let's use a heat map to visualize the relationship between Body Style vs Price

#use the grouped results
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()


# In[48]:


fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


# In[49]:


from scipy import stats


# In[50]:


pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# In[51]:


#Since the p-value is  <  0.001, the correlation between wheel-base and price is statistically significant, 
#although the linear relationship isn't extremely strong (~0.585).

pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  


# In[52]:


#Since the p-value is  <  0.001, the correlation between horsepower and price is statistically significant, 
#and the linear relationship is quite strong (~0.809, close to 1).

pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value ) 


# In[53]:


#Since the p-value is < 0.001, the correlation between highway-mpg and price is statistically significant, 
#and the coefficient of about -0.705 shows that the relationship is negative and moderately strong.


# In[54]:


# ANOVA
#The Analysis of Variance (ANOVA) is a statistical method used to test whether there are 
#significant differences between the means of two or more groups. ANOVA returns two parameters:
#F-test score: ANOVA assumes the means of all groups are the same, calculates how much the actual means deviate 
#from the assumption, and reports it as the F-test score. A larger score means there is a larger difference between the means.
#P-value: P-value tells how statistically significant our calculated score value is.

#Drive Wheels

grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test2.head(2)


# In[55]:


df_gptest


# In[56]:


#We can obtain the values of the method group using the method "get_group".
grouped_test2.get_group('4wd')['price']


# In[57]:


#We can use the function 'f_oneway' in the module 'stats' to obtain the F-test score and P-value


# In[58]:


# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val)   


# In[59]:


#This is a great result with a large F-test score showing a strong correlation and a P-value of almost 0 
#implying almost certain statistical significance. But does this mean all three tested groups are all this highly correlated?

#Let's examine them separately.


# In[60]:


#fwd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val )


# In[61]:


#4wd and rwd

f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])  
   
print( "ANOVA results: F=", f_val, ", P =", p_val)   


# In[62]:


#4wd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])  
   
print( "ANOVA results: F=", f_val, ", P =", p_val)   


# In[64]:


f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])  
 
print("ANOVA results: F=", f_val, ", P =", p_val)


# In[65]:


#print("ANOVA results: F=", f_val, ", P =", p_val)   
#We notice that ANOVA for the categories 4wd and fwd yields a high p-value > 0.1, 
#so the calculated F-test score is not very statistically significant. This suggests we can't reject the 
#assumption that the means of these two groups are the same, or, in other words, we can't conclude the difference 
#in correlation to be significant.


# In[ ]:


#Conclusion: Important Variables¶
#We now have a better idea of what our data looks like and which variables are important to take into 
#account when predicting the car price. We have narrowed it down to the following variables:

#Continuous numerical variables:

#Length
#Width
#Curb-weight
#Engine-size
#Horsepower
#City-mpg
#Highway-mpg
#Wheel-base
#Bore

#Categorical variables:
#Drive-wheels

#As we now move into building machine learning models to automate our analysis, 
#feeding the model with variables that meaningfully affect our target variable will improve our model's prediction performance.

