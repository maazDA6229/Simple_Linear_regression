#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[27]:


dataset = pd.read_csv("placement.csv")
dataset.head(3)


# In[28]:


plt.figure(figsize = (5,3))
sns.scatterplot(x = "cgpa", y = "package", data = dataset)
plt.show()


# In[29]:


dataset.isnull().sum()


# In[30]:


x = dataset[["cgpa"]]
y = dataset["package"]


# In[145]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 42 )


# In[146]:


from sklearn.linear_model import LinearRegression


# In[147]:


lr = LinearRegression()
lr.fit(x_train,y_train)


# In[148]:


# y = m*x+c


# In[149]:


lr.coef_  #value of m


# In[150]:


lr.intercept_ #value of c


# In[151]:


# y = 0.57425647*x-1.0270069374542108
    


# In[141]:


lr.score(x_test,y_test)*100


# In[153]:


lr.predict([[6.89]])


# In[152]:


0.57425647*6.89-1.0270069374542108


# In[ ]:





# In[157]:


y_prd = lr.predict(x)


# In[160]:


plt.figure(figsize = (5,4))
sns.scatterplot(x = "cgpa", y = "package", data = dataset)
plt.plot(dataset["cgpa"],y_prd, c = "red")
plt.legend(["org data","predict line"])
plt.savefig("predict.jpg")
plt.show()


# In[ ]:





# In[ ]:




