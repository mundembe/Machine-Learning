#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 22:27:54 2021

@author: masimba
"""
# <codecell>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

Dataset = pd.read_csv("Position_Salaries.csv")
X = Dataset.iloc[:, 1:-1].values
y = Dataset.iloc[:, -1].values

# <codecell>
# Train the Linear Reg model on the whole Dataset
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Train the Polynomial Regression model on the whole dataset
poly_feat = PolynomialFeatures(degree=4)    # matrix with X^1, X^2 ... X^n
X_poly = poly_feat.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)    # Model train using polynomial matrix of features

# Visualize Linear Regression results
plt.scatter (X, y, color="red")
plt.plot(X, lin_reg.predict(X), color="blue")
plt.title("Salary vs Position (Linear Regression)")
plt.xlabel("position")
plt.ylabel("Salary")
plt.show()

# Visualize Polynomial Linear Regression results
plt.scatter (X, y, color="red")
plt.plot(X, lin_reg_2.predict(X_poly), color="blue")
plt.title("Salary vs Position (Polynomial Regression)")
plt.xlabel("position")
plt.ylabel("Salary")
plt.show()

# Higher resolution visualisation Polynomial Linear Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter (X, y, color="red")
plt.plot(X, lin_reg_2.predict(poly_feat.fit_transform(X_grid)), color="blue")
plt.title("Salary vs Position (high-res Polynomial Regression)")
plt.xlabel("position")
plt.ylabel("Salary")
plt.show()

# Predict single observation using Linear reg
lin_reg.predict([[6.5]])    # The result from this is wrong (Very inaccurate)

# Predict single observation using Linear reg
lin_reg_2.predict(poly_feat.fit_transform([[6.5]]))