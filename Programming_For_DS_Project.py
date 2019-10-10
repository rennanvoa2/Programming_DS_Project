# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:12:24 2019

@author: renna
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
arrivals=pd.read_csv('International Arrivals.csv', header=2)
arrivals=arrivals[['Country Name', 'Country Code', '2007' ,'2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']]

metadata_country=pd.read_csv('Metadata_Country.csv')

df=arrivals.merge(metadata_country, on='Country Code', how='left')


#Set Country Name as Index
new_index = df['Country Name']
df.set_index(new_index,inplace=True )

#Create a column named Is_Country for later removing the "areas" like asia
df['Is_Country'] = df['Region'].notnull()

#Drop unnecessary columns
df.drop(['Country Name', 'Unnamed: 5', 'Region', 'IncomeGroup', 'SpecialNotes',
         'TableName'], inplace=True, axis=1)

#drop the 'areas'
df = df[df.Is_Country != False]

#Count number of itens in the DataFrame
df['Country Code'].count()
#Count not nulls in column 2008
df['2008'].count()

#count nulls in 2008
df['2008'].isnull().sum()

#drop rows with more than 10 NANs values
df = df.dropna(thresh=(len(df.columns) - 10))

#Check if the number of itens is correspondent after drop the rows
df['Country Code'].count()

#drop the Is_Country column becouse we dont need it anymore
df.drop('Is_Country', inplace=True, axis=1)


#fill the NaN values with the row mean
df2 = df.loc[:, '2008':'2017']
df2 = df2.apply(lambda row: row.fillna(row.mean()), axis=1)
for i in df2.columns:
    df[i] = df2[i]


df['Growth5ys']=(df['2017']/df['2012']-1)
df['Growth10ys']=(df['2017']/df['2007']-1)

df['Avg_In_Last_5_Years'] = df.loc[:,'2012':'2017'].mean(axis=1)
df['Avg_In_Last_10_Years'] = df.loc[:,'2008':'2017'].mean(axis=1)
