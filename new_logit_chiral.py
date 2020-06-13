import numpy as np
import csv
#import matplotlib.pyplot as plt
from sklearn import linear_model
from keras.utils import np_utils


# Read data from file (2atom-chiral.csv)
with open('2atoms-chiral.csv', 'r') as csvfile:
    dataReader = csv.reader(csvfile, delimiter=',')
    # Skip Header
    next(dataReader)
    chiral_data = []
    count = 0
    for row in dataReader:
        # print("line Num = %02d" % count)
        chiral_data.append([row[7], row[12]])
        count += 1



#X_chiral = np.asarray(chiral_data[1:])
X_chiral = np.asarray(chiral_data)
X_chiral = X_chiral.astype(np.float)

print('len=', len(X_chiral))

y_chiral = np.ones((len(X_chiral)))

print(X_chiral)
print(y_chiral)


# Read data from file (2atom-achiral.csv)
with open('2atoms-achiral.csv', 'r') as csvfile:
    dataReader = csv.reader(csvfile, delimiter=',')
    # Skip Header
    next(dataReader)
    chiral_data = []
    count = 0
    for row in dataReader:
        #    print "line Num = %02d" % count
        chiral_data.append([row[7], row[12]])
        count += 1


#X_achiral = np.asarray(chiral_data[1:])
X_achiral = np.asarray(chiral_data)
X_achiral = X_achiral.astype(np.float)

print('len=', len(X_achiral))

y_achiral = np.zeros((len(X_achiral)))

print(X_achiral)
print(y_achiral)

X = np.concatenate((X_chiral, X_achiral), axis=0)
y = np.concatenate((y_chiral, y_achiral), axis=0)
print('X=', X)
print('y=', y)

# Binary representation of the variables
a_categorical = np_utils.to_categorical(X[:,0]-1)
print('A=', a_categorical)


b_categorical = np_utils.to_categorical(X[:,1]-1)

print('B=', b_categorical)

#print(np.sum(b_categorical, axis=0))

# concatenate a_categorical and b_categorical
abxy = np.c_[a_categorical, b_categorical]
#abxy = np.c_[np.c_[a_categorical, b_categorical[:,8:]], X[:,3]/X[:,2]]
#abxy = np.c_[np.c_[a_categorical, b_categorical[:,8:]], X[:,2:3]]

print('AB=', abxy)

#print(np.sum(abxy, axis=0))

# logistic regression for categorical A and B
logreg = linear_model.LogisticRegression()

logreg.fit(abxy, y)

pred = logreg.predict(abxy)

#print('pred=', pred)

score = logreg.score(abxy, y)

print('Logistic Regression for Categorical Data (using all samples)')
print('Recognition Rates = ', score)

coeffs = logreg.coef_[0]
intercept = logreg.intercept_[0]

#print('Coeffs is ', coeffs)
print('Intercept is ', intercept)
print('Coeffs are')
for i in range(0, len(coeffs)):
    print(coeffs[i])


#===========================================================================
#### Reduce the number of achiral samples
# 1000 samples are randomly selected
sind = np.random.permutation(range(len(X_achiral)))
sind = sind[:1000]

X = np.concatenate((X_chiral, X_achiral[sind]), axis=0)
y = np.concatenate((y_chiral, y_achiral[sind]), axis=0)
print('X=', X)
print('y=', y)

# Binary representation of the variables
a_categorical = np_utils.to_categorical(X[:,0]-1)
print('A=', a_categorical)

print(np.sum(a_categorical, axis=0))

b_categorical = np_utils.to_categorical(X[:,1]-1)

print('B=', b_categorical)

print(np.sum(b_categorical, axis=0))

# concatenate a_categorical and b_categorical
abxy = np.c_[a_categorical, b_categorical]
#abxy = np.c_[np.c_[a_categorical, b_categorical[:,8:]], X[:,3]/X[:,2]]
#abxy = np.c_[np.c_[a_categorical, b_categorical[:,8:]], X[:,2:3]]

print('AB=', abxy)

#print(np.sum(abxy, axis=0))

# logistic regression for categorical A and B
logreg = linear_model.LogisticRegression()

logreg.fit(abxy, y)

pred = logreg.predict(abxy)

#print('pred=', pred)

score = logreg.score(abxy, y)

print('Logistic Regression for Categorical Data (1000 achiral samples are randomly selected)')
print('Recognition Rates = ', score)

coeffs = logreg.coef_[0]
intercept = logreg.intercept_[0]

#print('Coeffs is ', coeffs)
print('Intercept is ', intercept)
print('Coeffs are')
for i in range(0, len(coeffs)):
    print(coeffs[i])


#==============================================================================
# 2000 samples are randomly selected
sind = np.random.permutation(range(len(X_achiral)))
sind = sind[:2000]

X = np.concatenate((X_chiral, X_achiral[sind]), axis=0)
y = np.concatenate((y_chiral, y_achiral[sind]), axis=0)
print('X=', X)
print('y=', y)

# Binary representation of the variables
a_categorical = np_utils.to_categorical(X[:,0]-1)
print('A=', a_categorical)


b_categorical = np_utils.to_categorical(X[:,1]-1)

print('B=', b_categorical)

#print(np.sum(b_categorical, axis=0))

# concatenate a_categorical and b_categorical
abxy = np.c_[a_categorical, b_categorical]
#abxy = np.c_[np.c_[a_categorical, b_categorical[:,8:]], X[:,3]/X[:,2]]
#abxy = np.c_[np.c_[a_categorical, b_categorical[:,8:]], X[:,2:3]]

print('AB=', abxy)

#print(np.sum(abxy, axis=0))

# logistic regression for categorical A and B
logreg = linear_model.LogisticRegression()

logreg.fit(abxy, y)

pred = logreg.predict(abxy)

#print('pred=', pred)

score = logreg.score(abxy, y)

print('Logistic Regression for Categorical Data (2000 achiral samples are randomly selected)')
print('Recognition Rates = ', score)

coeffs = logreg.coef_[0]
intercept = logreg.intercept_[0]

#print('Coeffs is ', coeffs)
print('Intercept is ', intercept)
print('Coeffs are')
for i in range(0, len(coeffs)):
    print(coeffs[i])


