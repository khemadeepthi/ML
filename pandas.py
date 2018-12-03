#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 23:23:47 2018

@author: hema
"""

import pandas as pd
import os
from sklearn import tree

os.getcwd()
titanic_train = pd.read_csv("/Users/hema/Documents/DS_Learning/git/titanic/train.csv")
type(titanic_train)

titanic_train.shape
titanic_train.info()
titanic_train.describe()


titanic_train[2:4]
titanic_train[0:11]
titanic_train[0:11]['Fare']
titanic_train[:]['Embarked']
titanic_train['Embarked']
titanic_train[10:11]
titanic_train[9:10]

x_train = titanic_train[['Pclass','SibSp','Parch']]
y_train = titanic_train['Survived']
dt = tree.DecisionTreeClassifier()
dt.fit(x_train,y_train)

titanic_test = pd.read_csv("/Users/hema/Documents/DS_Learning/git/titanic/test.csv")
titanic_test['Survived'] = dt.predict(titanic_test[['Pclass','SibSp','Parch']])
titanic_test.to_csv("/Users/hema/Documents/DS_Learning/git/titanic/prediction1.csv", columns = ['PassengerId','Survived'], index = False)


