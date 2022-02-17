#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 19:28:13 2021

@author: masimba
"""
# <codecell>
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import Dataset
Dataset = pd.read_csv("Data.csv")
X = Dataset.iloc[:, :-1].values
y = Dataset.iloc[:, -1].values

# Impute missing values
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X[:, 1:] = imputer.fit_transform(X[:, 1:])

# Encode Features
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])],
                       remainder="passthrough")
X = np.array(ct.fit_transform(X))

# Encode Labels (dependant variable)
le = LabelEncoder()
y = le.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=1)

# Feature scaling
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
