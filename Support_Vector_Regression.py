#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 18:41:57 2022

@author: masimba
"""

# <codecell>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

Dataset = pd.read_csv("Position_Salaries.csv")
X = Dataset.iloc[:, 1:-1].values
y = Dataset.iloc[:, -1].values

## Scaling
y = y.reshape(len(y), 1)
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# <codecell>
regressor = SVR(kernel="rbf")
regressor.fit(X, y)

# Predict a single result
value = sc_X.transform([[6.5]])
y_pred = regressor.predict(value)
y_pred = sc_y.inverse_transform(y_pred)

inv_X = sc_X.inverse_transform(X)
inv_y = sc_y.inverse_transform(y)

# plot results (Higher definition)
plt.scatter(inv_X, inv_y, color="red")
y_pred_arr = regressor.predict(X)
plt.plot(inv_X, sc_y.inverse_transform(y_pred_arr), color="blue")
plt.show()

# Plot with higher resolution
X_range = np.arange(min(inv_X), max(inv_X), step=0.1)
X_range = X_range.reshape(len(X_range), 1)

plt.scatter(inv_X, inv_y, color="red")
scaled_X_range = sc_X.transform(X_range)
y_pred_smooth = regressor.predict(scaled_X_range)
plt.plot(X_range, sc_y.inverse_transform(y_pred_smooth), color="blue")
#plt.plot(X_range, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_range))), color = 'blue')
plt.show()