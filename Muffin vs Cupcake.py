#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Packages for data analysis

import numpy as np
import pandas as pd

from sklearn import svm

# Data visualization

import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale = 1.2)

# Inline matplotlib for jupyter Notebooks

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


recipes = pd.read_csv('muffin_vs_cupcake.csv')
print(recipes.head())


# In[4]:


# Plotting data

sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70});


# In[7]:


# Data pre-processing

type_label = np.where(recipes['Type']=='Muffin', 0, 1)
recipe_features = recipes.columns.values[1:].tolist()
recipe_features
ingredients = recipes[['Flour', 'Sugar']].values
print(ingredients)


# In[8]:


# Fit the model

model = svm.SVC(kernel='linear')
model.fit(ingredients, type_label)


# In[9]:


# Separating hyperplane

w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(30, 60)
yy = a * xx - (model.intercept_[0]) / w[1]

# Plot the parallels to the separating hyperplane that pass through the support vectors
b = model.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = model.support_vectors_[1]
yy_up = a * xx + (b[1] - a * b[0])


# In[13]:


sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')


# In[27]:


# Predict muffin or cupcake

def muffin_or_cupcake(flour, sugar):
    if(model.predict([[flour, sugar]]))==0:
        print('It is a MUFFIN!')
    else:
        print('It is a CUPCAKE')
        
    sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
    plt.plot(xx, yy, linewidth=2, color='black')
    plt.plot(flour, sugar, 'yo', markersize='20')


# In[28]:


muffin_or_cupcake(50, 20)


# In[34]:


muffin_or_cupcake(30, 50)


# In[ ]:




