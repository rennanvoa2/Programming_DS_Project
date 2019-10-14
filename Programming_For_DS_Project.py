import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
arrivals=pd.read_csv('International Arrivals.csv', header=2)
arrivals=arrivals[['Country Name', 'Country Code', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']]

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

#drop rows with more than 5 NANs values
arrivals_df = arrivals_df.dropna(thresh=(len(arrivals_df.columns) - 5))

#Check if the number of itens is correspondent after drop the rows
arrivals_df['Country Code'].count()

#drop the Is_Country column becouse we dont need it anymore
arrivals_df.drop('Is_Country', inplace=True, axis=1)


#fill the NaN values with the row mean
df2 = arrivals_df.loc[:, '2008':'2017']
df2 = df2.apply(lambda row: row.fillna(row.mean()), axis=1)
for i in df2.columns:
    arrivals_df[i] = df2[i]


#create feature Avarage in Last 10 Years
arrivals_df['Avg_Arrivals_10_Years'] = arrivals_df.loc[:,'2008':'2017'].mean(axis=1)



#Create the feature Growth in 10 years
arrivals_df['Growth10ys']=(arrivals_df['2017']/arrivals_df['2008']-1)



#_____________________________________________________________________________


receipts=pd.read_csv('Receipts.csv', header=2)
receipts=receipts[['Country Name', 'Country Code','2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']]

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

#drop rows with more than 5 NANs values
receipts_df = receipts_df.dropna(thresh=(len(receipts_df.columns) - 5))

#Check if the number of itens is correspondent after drop the rows
receipts_df['Country Code'].count()

#drop the Is_Country column becouse we dont need it anymore
receipts_df.drop('Is_Country', inplace=True, axis=1)


#fill the NaN values with the row mean
df2 = receipts_df.loc[:, '2008':'2017']
df2 = df2.apply(lambda row: row.fillna(row.mean()), axis=1)
for i in df2.columns:
    receipts_df[i] = df2[i]


#divided every value for 1000000. Now the scale is in milions of dolars
for i in range(2008,2018):
    receipts_df[str(i)] = receipts_df[str(i)].apply(lambda x: x / 1000000)


#create feature Avarage in Last 10 Years
receipts_df['Avg_Receipts_10_Years'] = receipts_df.loc[:,'2008':'2017'].mean(axis=1)


#Create the feature Growth in 10 years
receipts_df['Growth10ys']=(receipts_df['2017']/receipts_df['2008']-1)


total_number_weight = 2
growth_weight = 1
#____________________________________________________________
    
    
#sort by the best avarage arrivals in the last 10 years
better_arrivals = arrivals_df.sort_values('Avg_Arrivals_10_Years', ascending =False)

#get the best 40 countries
#better_arrivals = better_arrivals.iloc[0:40,:]

#divide each value of Growth in 10 years for the sum of the column
better_arrivals['% growth'] = better_arrivals['Growth10ys'] / better_arrivals['Growth10ys'].sum()

better_arrivals['% growth'].sum()

#divide each value of Avarage in 10 years for the sum of the column
better_arrivals['% avarage'] = better_arrivals['Avg_Arrivals_10_Years'] / better_arrivals['Avg_Arrivals_10_Years'].sum()


#Calculate the avarage between Growth and Avarage Numbers of Arrivals
better_arrivals['Growth x Avarage'] = (growth_weight * better_arrivals['% growth'] + (total_number_weight * better_arrivals['% avarage'])) / 4

#create a dataframe sorted by Growth X Avarage
Arrivals_in_growth_vs_arrivals = better_arrivals.sort_values('Growth x Avarage', ascending=False)

#____________________________________________________________--


#sort by the best avarage receipts in the last 10 years
better_receipts = receipts_df.sort_values('Avg_Receipts_10_Years', ascending =False)

#get the best 40 countries
#better_receipts = better_receipts.iloc[0:40,:]

#divide each value of Growth in 10 years for the sum of the column
better_receipts['% growth'] = better_receipts['Growth10ys'] / better_receipts['Growth10ys'].sum()

better_receipts['% growth'].sum()

#divide each value of Avarage in 10 years for the sum of the column
better_receipts['% avarage'] = better_receipts['Avg_Receipts_10_Years'] / better_receipts['Avg_Receipts_10_Years'].sum()


#Calculate the avarage between Growth and Avarage Numbers of Arrivals
better_receipts['Growth x Avarage'] = (growth_weight * better_receipts['% growth'] + (total_number_weight* better_receipts['% avarage'])) / 4


#create a dataframe sorted by Growth X Avarage
Receipts_in_growth_vs_receipt = better_receipts.sort_values('Growth x Avarage', ascending=False)


#________________________________________________________
