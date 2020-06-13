import numpy as np
from sklearn import linear_model
from keras.utils import np_utils

# read data from csv file
data = np.loadtxt("chiral.csv", delimiter=",", skiprows=1)

X = data[:,:-1]
y = data[:,-1]

#print('X=', X)
#print('y=', y)

# simple logistic regression for the original data
logreg = linear_model.LogisticRegression()

logreg.fit(X, y)

pred = logreg.predict(X)

#print('pred=', pred)

score = logreg.score(X, y)

print('Logistic Regression for the original Data')
print('Recognition Rate = ', score)

coeffs = logreg.coef_
intercept = logreg.intercept_

print('Coeffs is ', coeffs)
print('Intercept is ', intercept)


# Binary representation of the variables
a_categorical = np_utils.to_categorical(X[:,0]-1)
print(a_categorical)

print(np.sum(a_categorical, axis=0))

b_categorical = np_utils.to_categorical(X[:,1]-1)

print(b_categorical)

print(np.sum(b_categorical, axis=0))

# concatenate a_categorical and b_categorical
abxy = np.c_[a_categorical, b_categorical[:,8:]]
#abxy = np.c_[np.c_[a_categorical, b_categorical[:,8:]], X[:,3]/X[:,2]]
#abxy = np.c_[np.c_[a_categorical, b_categorical[:,8:]], X[:,2:3]]

print(abxy)

print(np.sum(abxy, axis=0))

# logistic regression for categorical A and B
logreg = linear_model.LogisticRegression()

logreg.fit(abxy, y)

pred = logreg.predict(abxy)

#print('pred=', pred)

score = logreg.score(abxy, y)

print('Logistic Regression for Categorical Data')
print('Recognition Rates = ', score)

coeffs = logreg.coef_
intercept = logreg.intercept_

print('Coeffs is ', coeffs)
print('Intercept is ', intercept)




