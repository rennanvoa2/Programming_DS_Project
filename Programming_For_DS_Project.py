# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 12:34:35 2019

@author: renna
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#set the path of the file
path = 'C:/FILES/API_ST.INT.ARVL_DS2_en_csv_v2_103871.csv'

#Removing the columns(Indicator Name, Indicator Code and Years between 1960 
#and 1994, 2018, 2019 that doesn't have any usefull data)
df = pd.read_csv(path, sep=',', skip_blank_lines=True, skiprows=4)
columns_to_drop = ['Indicator Name', 'Indicator Code']
year = 1960
while year <= 1994:
    columns_to_drop.append(str(year))
    year += 1
columns_to_drop.append('2018')
dataset = df.drop(columns_to_drop, axis=1)
dataset.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)


