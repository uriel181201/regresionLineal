from sklearn import linear_model
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
anuncio = pd.read_csv('advertising.csv')

# Creamos las variables:
feature_cols = ['Daily Time Spent on Site']
X = anuncio[feature_cols]
y = anuncio['Clicked on Ad']


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)



model = LinearRegression(fit_intercept=True)
LinearRegression()

model.fit(Xtrain, ytrain) # fitting the model


pickle.dump(model, open('anuncios.pkl','wb')) # save the model

print(model.predict([[15]]))  # format of input
print(f'score: {model.score(X, y)}')
