#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 07:52:45 2022

@author: masimba
"""
# <codecell>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

Dataset = pd.read_csv("Position_Salaries.csv")
X = Dataset.iloc[:, 1:-1].values
y = Dataset.iloc[:, -1].values

# <codecell>
# Train the model
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)

# Plot high def
X_range = np.arange(min(X), max(X), 0.1)
X_range = X_range.reshape(len(X_range), 1)
plt.scatter(X, y, color="red")
plt.plot(X_range, regressor.predict(X_range), color="blue")
plt.title("Position vs Salary (Random Forest)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()