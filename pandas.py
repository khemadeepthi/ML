#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 23:23:47 2018

@author: hema
"""

import pandas as pd
import os

os.getcwd()
titanic_train = pd.read_csv("/Users/hema/Documents/DS_Learning/git/titanic/train.csv")
type(titanic_train)

titanic_train.shape
titanic_train.info()
titanic_train.describe()


titanic_train[2:4]
titanic_train[0:11]
titanic_train[0:11]['Fare']
