# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x =LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

# Avoiding the dummy variable trap
x = x[:, 1:]

# Splitting the data set into training se and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Fitting the regression model to the training set
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)

# Predicting the result
y_pred = linear_reg.predict(x_test)

import sklearn.metrics
sklearn.metrics.r2_score(y_test, y_pred)

print(np.array([y_pred, y_test]))

# Predicting the profit of a new company added to the database
new_data = np.array([1, 40000000, 32000000, 50000000])
y_profit = linear_reg.predict(new_data)
y_profit

print('This is a child branch')
