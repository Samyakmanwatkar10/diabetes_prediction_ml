# Import Dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm 
from sklearn.metrics import accuracy_score

# Data Collection and Analysis
diabetes_dataset=pd.read_csv('diabetes.csv')
# print(diabetes_dataset.head())
# print(diabetes_dataset.shape)
# print(diabetes_dataset['Outcome'].value_counts())   #0-500, 1-268
# print(diabetes_dataset.groupby('Outcome').mean())

# Separating data and labels
X=diabetes_dataset.drop(columns='Outcome', axis=1)
Y=diabetes_dataset['Outcome']

# Data Standardization 
scaler=StandardScaler()
X=scaler.fit_transform(X)

# train test split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
# print(X.shape, X_train.shape, X_test.shape)

# Training the model
classifier=svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Model Evaluation
# accuracy score on training data
X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data: ', training_data_accuracy)

# accuracy score on test data
X_test_prediction=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction, Y_test)
print("Accuracy on test data: ", test_data_accuracy)
