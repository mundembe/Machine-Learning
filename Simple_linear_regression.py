#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 12:47:01 2022

@author: masimba
"""
# <codecell>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

Dataset = pd.read_csv("Salary_Data.csv")
X = Dataset.iloc[:, :-1].values
y = Dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=1)

# <codecell>
# Train simple linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict Test results
y_prediction = regressor.predict(X_test)

# Plot training set results
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train))
plt.title("Salary vs Experience (Training set)")
plt.xlabel("years of experience")
plt.ylabel("Salary")
plt.show()

# Plot test set results
plt.scatter(X_test, y_test, color="red")
plt.plot(X_test, y_prediction, color="blue")
plt.title("Salary vs Experience (Prediction)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
