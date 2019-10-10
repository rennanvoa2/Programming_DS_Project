import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
arrivals=pd.read_csv('International Arrivals.csv', header=2)
arrivals=arrivals[['Country Name', 'Country Code', '2007' ,'2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']]

metadata_country=pd.read_csv('Metadata_Country.csv')

arrivals_df=arrivals.merge(metadata_country, on='Country Code', how='left')


#Set Country Name as Index
new_index = arrivals_df['Country Name']
arrivals_df.set_index(new_index,inplace=True )

#Create a column named Is_Country for later removing the "areas" like asia
arrivals_df['Is_Country'] = arrivals_df['Region'].notnull()

#Drop unnecessary columns
arrivals_df.drop(['Country Name', 'Unnamed: 5', 'Region', 'IncomeGroup', 'SpecialNotes',
         'TableName'], inplace=True, axis=1)

#drop the 'areas'
arrivals_df = arrivals_df[arrivals_df.Is_Country != False]

#Count number of itens in the DataFrame
arrivals_df['Country Code'].count()
#Count not nulls in column 2008
arrivals_df['2008'].count()

#count nulls in 2008
arrivals_df['2008'].isnull().sum()

#drop rows with more than 10 NANs values
arrivals_df = arrivals_df.dropna(thresh=(len(arrivals_df.columns) - 10))

#Check if the number of itens is correspondent after drop the rows
arrivals_df['Country Code'].count()

#drop the Is_Country column becouse we dont need it anymore
arrivals_df.drop('Is_Country', inplace=True, axis=1)


#fill the NaN values with the row mean
df2 = arrivals_df.loc[:, '2007':'2017']
df2 = df2.apply(lambda row: row.fillna(row.mean()), axis=1)
for i in df2.columns:
    arrivals_df[i] = df2[i]


arrivals_df['Growth5ys']=(arrivals_df['2017']/arrivals_df['2012']-1)
arrivals_df['Growth10ys']=(arrivals_df['2017']/arrivals_df['2007']-1)

arrivals_df['Avg_In_Last_5_Years'] = arrivals_df.loc[:,'2012':'2017'].mean(axis=1)
arrivals_df['Avg_In_Last_10_Years'] = arrivals_df.loc[:,'2008':'2017'].mean(axis=1)


#_____________________________________________________________________________


receipts=pd.read_csv('Receipts.csv', header=2)
receipts=receipts[['Country Name', 'Country Code', '2007' ,'2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']]

metadata_country=pd.read_csv('Metadata_Country.csv')

receipts_df=receipts.merge(metadata_country, on='Country Code', how='left')


#Set Country Name as Index
new_index = receipts_df['Country Name']
receipts_df.set_index(new_index,inplace=True )

#Create a column named Is_Country for later removing the "areas" like asia
receipts_df['Is_Country'] = receipts_df['Region'].notnull()

#Drop unnecessary columns
receipts_df.drop(['Country Name', 'Unnamed: 5', 'Region', 'IncomeGroup', 'SpecialNotes',
         'TableName'], inplace=True, axis=1)

#drop the 'areas'
receipts_df = receipts_df[receipts_df.Is_Country != False]

#Count number of itens in the DataFrame
receipts_df['Country Code'].count()
#Count not nulls in column 2008
receipts_df['2008'].count()

#count nulls in 2008
receipts_df['2008'].isnull().sum()

#drop rows with more than 10 NANs values
receipts_df = receipts_df.dropna(thresh=(len(receipts_df.columns) - 10))

#Check if the number of itens is correspondent after drop the rows
receipts_df['Country Code'].count()

#drop the Is_Country column becouse we dont need it anymore
receipts_df.drop('Is_Country', inplace=True, axis=1)


#fill the NaN values with the row mean
df2 = receipts_df.loc[:, '2007':'2017']
df2 = df2.apply(lambda row: row.fillna(row.mean()), axis=1)
for i in df2.columns:
    receipts_df[i] = df2[i]


receipts_df['Growth5ys']=(receipts_df['2017']/receipts_df['2012']-1)
receipts_df['Growth10ys']=(receipts_df['2017']/receipts_df['2007']-1)

receipts_df['Avg_In_Last_5_Years'] = receipts_df.loc[:,'2012':'2017'].mean(axis=1)
receipts_df['Avg_In_Last_10_Years'] = receipts_df.loc[:,'2008':'2017'].mean(axis=1)
