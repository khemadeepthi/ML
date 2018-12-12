#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 20:29:33 2018

@author: hema
"""

import pandas as pd
import numpy as np

titanic_train = pd.read_csv("/Users/hema/Documents/DS_Learning/git/titanic/train.csv")
titanic_train.info()

# removing unwanted features 
titanic_refined = titanic_train.drop(['PassengerId','Name','Ticket','Fare','Cabin'], axis=1)
titanic_refined.info()

# calculate mean of age and fill missing age values
# titanic_refined['Age'][titanic_refined['Age'].isnull()] = titanic_refined['Age'].mean()
titanic_refined['Age']= np.where(titanic_refined['Age'].isnull(), titanic_refined['Age'].mean(), titanic_refined['Age'])

# calculate mode of Embarked and fill missing Embarked values
titanic_refined['Embarked']= np.where(titanic_refined['Embarked'].isnull(), titanic_refined['Embarked'].mode(), titanic_refined['Embarked'])
titanic_refined.info()

titanic_refined = pd.get_dummies(titanic_refined, columns = ['Embarked','Pclass','Sex'])
