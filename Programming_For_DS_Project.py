

#Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Define the URI from the github files (Number of Arrivels and Income)
income_uri = 'https://raw.githubusercontent.com/rennanvoa2/Programming_DS_Project/master/Income.csv?token=AGBCKJR4XZIDLPKEMDYI5UK5VXNQW'
arrival_uri = 'https://raw.githubusercontent.com/rennanvoa2/Programming_DS_Project/master/International%20Arrivals.csv?token=AGBCKJWXYPAEJDA36VRCGV25VXLRI'

#Read the Arrivals CSV
arrivals=pd.read_csv(arrival_uri, header=2)

#Select the columns with usefull data
arrivals=arrivals[['Country Name', 'Country Code', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']]

#Load metadata CSV
metadata_country=pd.read_csv('Metadata_Country.csv')

#Merge Arrivals CSV with Metadata CSV
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


#Load income CSV
income=pd.read_csv(income_uri, header=2)

#Select the columns with usefull data
income=income[['Country Name', 'Country Code','2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']]

#load Metadata CSV
metadata_country=pd.read_csv('Metadata_Country.csv')

#Merge income CSV and Metadata CSV
income_df=income.merge(metadata_country, on='Country Code', how='left')


#Set Country Name as Index
new_index = income_df['Country Name']
income_df.set_index(new_index,inplace=True )

#Create a column named Is_Country for later removing the "areas" like asia
income_df['Is_Country'] = income_df['Region'].notnull()

#Drop unnecessary columns
income_df.drop(['Country Name', 'Unnamed: 5', 'Region', 'IncomeGroup', 'SpecialNotes',
         'TableName'], inplace=True, axis=1)

#drop the 'areas'
income_df = income_df[income_df.Is_Country != False]

#Count number of itens in the DataFrame
income_df['Country Code'].count()

#Count not nulls in column 2008
income_df['2008'].count()

#count nulls in 2008
income_df['2008'].isnull().sum()

#drop rows with more than 5 NANs values
income_df = income_df.dropna(thresh=(len(income_df.columns) - 5))

#Check if the number of itens is correspondent after drop the rows
income_df['Country Code'].count()

#drop the Is_Country column becouse we dont need it anymore
income_df.drop('Is_Country', inplace=True, axis=1)


#fill the NaN values with the row mean
df2 = income_df.loc[:, '2008':'2017']
df2 = df2.apply(lambda row: row.fillna(row.mean()), axis=1)
for i in df2.columns:
    income_df[i] = df2[i]


#divided every value for 1000000. Now the scale is in milions of dolars
for i in range(2008,2018):
    income_df[str(i)] = income_df[str(i)].apply(lambda x: x / 1000000)


#create feature Avarage in Last 10 Years
income_df['Avg_Income_10_Years'] = income_df.loc[:,'2008':'2017'].mean(axis=1)


#Create the feature Growth in 10 years
income_df['Growth10ys']=(income_df['2017']/income_df['2008']-1)


total_number_weight = 1
growth_weight = 1

#____________________________________________________________
    
    
#sort by the best avarage arrivals in the last 10 years
better_arrivals = arrivals_df.sort_values('Avg_Arrivals_10_Years', ascending =False)


#divide each value of Growth in 10 years for the sum of the column
better_arrivals['% growth'] = better_arrivals['Growth10ys'] / better_arrivals['Growth10ys'].sum()

better_arrivals['% growth'].sum()

#divide each value of Avarage in 10 years for the sum of the column
better_arrivals['% avarage'] = better_arrivals['Avg_Arrivals_10_Years'] / better_arrivals['Avg_Arrivals_10_Years'].sum()


#Calculate the avarage between Growth and Avarage Numbers of Arrivals
better_arrivals['Growth x Avarage'] = (growth_weight * better_arrivals['% growth'] +
               (total_number_weight * better_arrivals['% avarage'])) / (total_number_weight + growth_weight)

#create a dataframe sorted by Growth X Avarage
Arrivals_in_growth_vs_arrivals = better_arrivals.sort_values('Growth x Avarage', ascending=False)

#____________________________________________________________--


#sort by the best avarage income in the last 10 years
better_income = income_df.sort_values('Avg_Income_10_Years', ascending =False)

#get the best 40 countries
#better_income = better_income.iloc[0:40,:]

#divide each value of Growth in 10 years for the sum of the column
better_income['% growth'] = better_income['Growth10ys'] / better_income['Growth10ys'].sum()

better_income['% growth'].sum()

#divide each value of Avarage in 10 years for the sum of the column
better_income['% avarage'] = better_income['Avg_Income_10_Years'] / better_income['Avg_Income_10_Years'].sum()


#Calculate the avarage between Growth and Avarage Numbers of Arrivals
better_income['Growth x Avarage'] = (growth_weight * better_income['% growth'] +
               (total_number_weight* better_income['% avarage'])) / (total_number_weight + growth_weight)


#create a dataframe sorted by Growth X Avarage
income_in_growth_vs_income = better_income.sort_values('Growth x Avarage', ascending=False)

#Drop Belarus, its an outlier in Arrivals Dataset
Arrivals_in_growth_vs_arrivals = Arrivals_in_growth_vs_arrivals.drop(['Belarus'])

#Drop Myanmar, its an outlier in income Dataset
income_in_growth_vs_income = income_in_growth_vs_income.drop(['Myanmar'])

#________________________________________________________

#get the best 10 results in Arrivals
arrivals_plot = Arrivals_in_growth_vs_arrivals.iloc[0:10,:]

#get the best 10 results in income
income_plot = income_in_growth_vs_income.iloc[0:10,:]


# plotting the points 
plt.bar(arrivals_plot.index.values, arrivals_plot['Growth x Avarage'], color='blue')

#Resize the figure to 50x20
plt.rcParams['figure.figsize'] = (50,20)

 
# naming the x axis 
plt.xlabel('Countries') 

# naming the y axis 
plt.ylabel('% in Growth x % in Number') 
  
# giving a title
plt.title("Arrival's") 

#Save it in a png file for better view
plt.savefig('Arrivals.png')

# function to show the plot 
plt.show() 


#________________________________________________________

# plotting the points 
plt.bar(income_plot.index.values, income_plot['Growth x Avarage'], color='orange')

#Resize the figure to 50x20
plt.rcParams['figure.figsize'] = (50,20)

# naming the x axis 
plt.xlabel('Countries') 

# naming the y axis 
plt.ylabel('% in Growth x % in Number') 
  
# giving a title 
plt.title('Income') 

#Save it in a png file for better view
plt.savefig('Income.png')

# function to show the plot 
plt.show() 


#________________________________________________________

#Create a list with unique countries in arrivals_plot + income_plot
label_array = list(set(list(arrivals_plot.index.values)+ list(income_plot.index.values)))


#Create a dataframe with the sum of top arrivals + top income
Arrivals_plus_income = pd.DataFrame(index=label_array)
Arrivals_plus_income['Arrivals GxA'] = better_arrivals['Growth x Avarage']
Arrivals_plus_income['Income GxA'] = better_income['Growth x Avarage']

#sort the DataFrame by income
Arrivals_plus_income = Arrivals_plus_income.sort_values('Income GxA', ascending =False)



#Plot  income Values on a Bar chart
plt.bar(Arrivals_plus_income.index.values, Arrivals_plus_income['Income GxA'], width=0.5,
        label='Income', color='orange')

#Plot  Arrivals Values on a Bar chart
plt.bar(Arrivals_plus_income.index.values, Arrivals_plus_income['Arrivals GxA'], width=0.5,
        label='Arrival', bottom=Arrivals_plus_income['Income GxA'], color='blue')

#Resize the figure to 100x20
plt.rcParams['figure.figsize'] = (100,20)

# naming the x axis 
plt.xlabel('Top Countries') 

# naming the y axis 
plt.ylabel('Income & Arrivals (Growth x Avarage)') 
  
# giving a title
plt.title('Growth x Total Number of Arrivals') 

#Put legend
plt.legend()

#Save it in a png file for better view
plt.savefig('Arrivals x Income.png')


# function to show the plot 
plt.show() 
    
#_______________________
