#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 21:44:05 2022

@author: masimba
"""

# <codecell>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
# cell 2
from sklearn.linear_model import LinearRegression

Dataset = pd.read_csv("50_Startups.csv")
X = Dataset.iloc[:, :-1].values
y = Dataset.iloc[:, -1].values

# Encode features (countries)
ct = ColumnTransformer(transformers=[("encode", OneHotEncoder(), [3])],
                       remainder="passthrough")
X = np.array(ct.fit_transform(X))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=1)
# <codecell>
# Train model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate( (y_test.reshape(len(y_test), 1),
                       y_pred.reshape(len(y_pred), 1) ),
                     axis=1) )
