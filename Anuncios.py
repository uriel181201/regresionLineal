#!/usr/bin/env python
# coding: utf-8

# Se importan las librerias necesarias

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Cargamos el dataset


# In[39]:


anuncio = pd.read_csv("advertising.csv")
anuncio.info()


# In[40]:


anuncio.head()


# 
# 
# EL DATASET DESCRIBE EL COMPORTAMIENTO DE LOS USUARIOS QUE PERMANECEN EN UN SITIO WEB (EN HORAS) Y LAS VECES QUE DAN CLIC A UN ANUNCIO

# Visualizamos los datos a traves de distintas gráficas utilizando pair plot

# In[41]:


sns.pairplot(anuncio);


# Visualizamos la matriz de correlación con un mapa de calor 

# In[42]:


sns.heatmap(anuncio.corr(), annot = True, vmin = -1, cmap = 'Blues');


# Vamos a utilizar la variable Daily Time Spent on Site y Clicked on Ad que son las que muestran mejor el comportamiento de los datos. Con el metodo regplot() se mostrará el grafico de la regresión lineal

# In[43]:


sns.regplot(x = "Daily Time Spent on Site",
            y = "Clicked on Ad",
            data = countries,
            dropna = True)
plt.xlim(-10, 200)
plt.ylim(bottom=0)


# In[44]:


from sklearn.linear_model import LinearRegression


# In[45]:


model = LinearRegression(fit_intercept=True)


# In[46]:


LinearRegression()


# Creamos la matriz de features  y el vector target para ajustar nuestro modelo de regresión lineal simple.

# In[64]:


# Creamos las variables:
feature_cols = ['Daily Time Spent on Site']
X = anuncio[feature_cols]
y = anuncio['Clicked on Ad']

# Corroboramos la forma y tipo de cada una:
print('Shape X:', X.shape)
print('Type X:', type(X))
print('Shape y:', y.shape)
print('Type y:', type(y))


# Se utiliza la función train_test_split() para que el modelo aprenda a través de los datos y el entrenamiento para posteriormente hacer la predicción

# In[49]:


from sklearn.model_selection import train_test_split


# In[50]:


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state = 1)


# In[51]:


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)


# Utilizamos el método fit para que el usuario explore los atributos de la clase 
# 

# In[52]:


model.fit(Xtrain, ytrain)


# In[53]:


LinearRegression()


# In[63]:


model.intercept_ 


# In[62]:


model.coef_ 


# Posteriormente se predicen etiquetas para datos desconocidos

# In[56]:


test = 20
model.intercept_ + model.coef_ * test


# In[61]:


test_sklearn = np.array(test).reshape(-1,1) 
test_sklearn


# In[58]:


ypred = model.predict(Xtest)
ypred


# Se evalua el perfomance del modelo comparando las predicciones con los valores reales

# In[59]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[60]:


print('MAE:', mean_absolute_error(ytest, ypred).round(2))
print('MSE:', mean_squared_error(ytest, ypred).round(2))
print('RMSE:', np.sqrt(mean_squared_error(ytest, ypred)).round(2))
print('R2:', r2_score(ytest, ypred).round(2))

