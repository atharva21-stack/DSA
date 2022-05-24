# Importing the libraries
import numpy as np # used for handling numbers
import pandas as pd # used for handling the dataset

dataset = pd.read_csv('data.csv') # to import the dataset into a variable

print(dataset)

from sklearn.impute import SimpleImputer  # used for handling missing data

# 'np.nan' signifies that we are targeting missing values
# and the strategy we are choosing is replacing it with 'mean'

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(dataset.iloc[:, 1:3])
dataset.iloc[:, 1:3] = imputer.transform(dataset.iloc[:, 1:3])  

# print the dataset
print(dataset)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder  #OneHot Encoding consists of turning the country column into three separate columns, # each column consists of 0s and 1s. 

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# [0] signifies the index of the column we are appliying the encoding on
data = pd.DataFrame(ct.fit_transform(dataset))

print(data)
from sklearn.preprocessing import LabelEncoder # used for encoding categorical data

le = LabelEncoder()
data.iloc[:,-1] = le.fit_transform(data.iloc[:,-1])
# 'data.iloc[:,-1]' is used to select the column that we need to be encoded

print(data)

from sklearn.preprocessing import MinMaxScaler  
# When we normalize the dataset it brings the value of all the features between 0 and 1 
# so that all the columns are in the same range

scaler = MinMaxScaler()
data = pd.DataFrame(scaler.fit_transform(data))

print(data)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
# .values function coverts the data into arrays

print("Independent Variable\n")
print(X)
print("\nDependent Variable\n")
print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
#'test_size=0.2' means 20% test data and 80% train data


print("X_train\n")
print(X_train)
print("\nX_test\n")
print(X_test)
print("y_train\n")
print(y_train)
print("\ny_test\n")
print(y_test)

