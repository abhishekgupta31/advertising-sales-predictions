#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


# In[2]:


df = pd.read_csv('Advertising.csv')


# In[3]:


df


# In[4]:


df_n=df.drop(['Unnamed: 0'],axis=1)


# In[5]:


df_n.shape


# In[6]:


df_n.info()


# In[7]:


df_n.describe()


# In[8]:


df_n.isnull().sum()


# In[9]:


df.skew()


# In[10]:


print(sns.boxplot(df_n['TV']))


# In[11]:


print(sns.boxplot(df['radio']))


# In[12]:


print(sns.boxplot(df['newspaper']))


# In[13]:


for col in df_n.columns:
    plt.figure(figsize=(5,5))
    sns.distplot(df[col])


# In[14]:


sns.pairplot(df_n, x_vars=['TV', 'newspaper', 'radio'], y_vars='sales', height=4, aspect=1, kind='scatter')
plt.show()


# In[15]:


sns.jointplot(x=df_n[col],y=df_n['sales'])


# In[16]:


sns.jointplot(x=df_n['TV'],y=df_n['sales'])


# In[17]:


sns.jointplot(x=df_n['radio'],y=df_n['sales'])


# In[18]:


sns.jointplot(x=df_n['newspaper'],y=df_n['sales'])


# In[19]:


sns.heatmap(df_n.isnull())


# In[20]:


df_n.corr()


# In[21]:


sns.heatmap(df_n.corr(),annot=True)


# In[22]:


q = df_n.sales.describe()
print(q)
IQR    = q['75%'] - q['25%']
Upper  = q['75%'] + 1.5 * IQR
Lower  = q['25%'] - 1.5 * IQR
print("the upper and lower outliers are {} and {}".format(Upper,Lower))


# In[23]:


x = df_n.drop('sales',axis=1)
y = df_n[['sales']]


# In[24]:


print(x.shape)
print(y.shape)


# In[25]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=101)


# In[26]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[27]:


lr = LinearRegression()
lr.fit(x_train,y_train)


# In[28]:


lr.intercept_


# In[29]:


lr.coef_


# In[30]:


pred = lr.predict(x_test)
pred


# In[31]:


from sklearn.metrics import r2_score


# In[32]:


r2_score(y_test,pred)


# In[33]:


import statsmodels.api as sm
x_train_sm = sm.add_constant(x_train)


# In[34]:


lre = sm.OLS(y_train,x_train_sm).fit()
lre.summary()


# In[35]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['Features'] = x_train.columns
vif['VIF'] = [variance_inflation_factor(x_train.values, i) for i in range(x_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[36]:


x_train_sm=sm.add_constant(x_train[['TV','radio']])
x_train_sm


# In[37]:


lre = sm.OLS(y_train,x_train_sm).fit()
lre


# In[38]:


lre.summary()


# In[39]:


lnr= LinearRegression()


# In[40]:


from sklearn.feature_selection import RFE
rfe=RFE(lnr,2.0)


# In[41]:


rfe.fit(x_train,y_train)


# In[42]:


rfe.support_


# In[43]:


a=x_train.columns[rfe.support_]
a


# In[44]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()


# In[45]:


x_train=sc.fit_transform(x_train[a])


# In[46]:


x_test=sc.fit_transform(x_test[a])


# In[47]:


lre=LinearRegression()


# In[48]:


lre.fit(x_train,y_train)


# In[51]:


lre.intercept_


# In[52]:


lre.coef_


# In[55]:


pred= lre.predict(x_test)


# In[56]:


r2_score(y_test,pred)


# In[57]:


plt.scatter(y_test,pred)
plt.show()


# In[58]:


sns.distplot(y_test-pred)
plt.show()


# In[ ]:




