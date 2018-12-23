#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 20:31:18 2018

@author: hema
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection

titanic_train = pd.read_csv("/Users/hema/Documents/DS_Learning/git/titanic/train.csv")

titanic_train.info()

titanic_train['Embarked'] = np.where(titanic_train['Embarked'].isnull(), titanic_train['Embarked'].mode(), titanic_train['Embarked'])

titanic_train_1Hot = pd.get_dummies(titanic_train,columns=['Pclass','Sex','Embarked'])
titanic_train_1Hot.info()
titanic_train_filtered = titanic_train_1Hot.drop(axis = 1, columns=['Name','PassengerId','Age','Cabin','Ticket','Fare'])
titanic_train_filtered.info()

x_train = titanic_train_filtered[['SibSp','Parch','Pclass_1','Pclass_2','Pclass_3','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S']]
y_train = titanic_train_filtered[['Survived']]

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
cv_scores = model_selection.cross_val_score(dt, x_train, y_train, cv=10)
print(cv_scores)
print(cv_scores.mean())

dt1= DecisionTreeClassifier(criterion = "entropy")
dt1.fit(x_train, y_train)
cv_scores1 = model_selection.cross_val_score(dt1, x_train, y_train, cv=10)
print(cv_scores1.mean())

dt2 = DecisionTreeClassifier(criterion = "entropy", max_depth = 5)
dt2.fit(x_train, y_train)
cv_scores2 = model_selection.cross_val_score(dt2, x_train, y_train, cv=10)
print(cv_scores2.mean())

dt3 = DecisionTreeClassifier()
dt3.fit(x_train, y_train)
param_grid = {'criterion':['entropy','gini'], 'max_depth': [5,7,10]}
dt3_grid = model_selection.GridSearchCV(dt3, param_grid, n_jobs=3, cv=10)
dt3_grid.fit(x_train, y_train)
dt3_grid.cv_results_
dt3_grid.best_score_
dt3_grid.best_estimator_
print(dt3_grid.best_score_)
print(dt3_grid.score(x_train, y_train))




