#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 22:53:09 2022

@author: masimba
"""
# <codecell>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

Dataset = pd.read_csv("Position_Salaries.csv")
X = Dataset.iloc[:, 1:-1].values
y = Dataset.iloc[:, -1].values

# <codecell>
# Train model
regressor = DecisionTreeRegressor(random_state=1)
regressor.fit(X, y)

# Predict single value
pred = regressor.predict([[6.5]])

# Plot high res results
X_range = np.arange(min(X), max(X), 0.1)
X_range = X_range.reshape(len(X_range), 1)
plt.scatter(X, y, color="red")
plt.plot(X_range, regressor.predict(X_range), color="blue")
plt.title("Decision-tree-regression")
plt.show()
