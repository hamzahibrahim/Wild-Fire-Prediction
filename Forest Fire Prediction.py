#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Data.csv')
print('\nThe first 5 rows of the data',df.head())
print()
print('The empty values: \n', df.isnull().sum().sum())
print()
print(df.describe())

df["CLASS"].value_counts().plot.bar(color='orange')
plt.title('No Fire VS Fire')
plt.show()

print()
print('No Fire:', df['CLASS'].value_counts()[0])
print('Fire:', df['CLASS'].value_counts()[1])
print()

df = df.reindex(np.random.permutation(df.index))
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
print('X Shape:', X.shape)
print('y shape:', y.shape)

# Encoding the class column [1: No fire, 0: Fire]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# LogisticRegression
from sklearn.linear_model import LogisticRegression
firstModel = LogisticRegression(
    solver="liblinear", max_iter=120, random_state=0)
firstModel.fit(X_train, y_train)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(firstModel, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Confusion matrix for the LogisticRegression')
plt.show()

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = firstModel.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix: \n', cm)
print('\nAccuracy score: ', accuracy_score(y_test, y_pred))

# Decession Tree
from sklearn.tree import DecisionTreeClassifier
secondModel = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
secondModel.fit(X_train, y_train)

plot_confusion_matrix(secondModel, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Confusion matrix for the DecisionTree')
plt.show()

y_pred2 = secondModel.predict(X_test)
cm2 = confusion_matrix(y_test, y_pred2)
print('Confusion matrix for the DecisionTree: \n', cm2)
print('\nDecisionTree Accuracy score: ', accuracy_score(y_test, y_pred2))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
thirdModel = RandomForestClassifier(n_estimators = 2000, criterion='entropy')
thirdModel.fit(X_train, y_train)

plot_confusion_matrix(thirdModel, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Confusion matrix for the Random Forest')
plt.show()

y_pred3 = thirdModel.predict(X_test)
cm3 = confusion_matrix(y_test, y_pred3)
print('Confusion matrix for the Random Forest: \n', cm3)
print('\nRandom Forest Accuracy score: ', accuracy_score(y_test, y_pred3))

#Kernal Svm
from sklearn.svm import SVC
forthModel = SVC(C=3.0)
forthModel.fit(X_train, y_train)

plot_confusion_matrix(forthModel, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Confusion matrix for the Kernal SVM')
plt.show()

y_pred4 = forthModel.predict(X_test)
cm4 = confusion_matrix(y_test, y_pred4)
print('Confusion matrix for the Kernal SVM: \n', cm4)
print('\nKernal SVM Accuracy score: ', accuracy_score(y_test, y_pred4))

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
fifthModel = GaussianNB()
fifthModel.fit(X_train, y_train)

plot_confusion_matrix(fifthModel, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Confusion matrix for the Naive Bayes')
plt.show()

y_pred5 = fifthModel.predict(X_test)
cm5 = confusion_matrix(y_test, y_pred5)
print('Confusion matrix for the Naive Bayes: \n', cm5)
print('\nNaive Bayes Accuracy score: ', accuracy_score(y_test, y_pred5))

#See the models
print('===========================')
print('\nLogistic RegressionAccuracy score: ', accuracy_score(y_test, y_pred))
print('DecisionTree Accuracy score: ', accuracy_score(y_test, y_pred2))
print('Random Forest Accuracy score: ', accuracy_score(y_test, y_pred3))
print('Kernal SVM Accuracy score: ', accuracy_score(y_test, y_pred4))
print('Naive Bayes Accuracy score: ', accuracy_score(y_test, y_pred5))
print('\n===========================')

import time
time.sleep(5)
