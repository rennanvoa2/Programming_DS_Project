
#Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from copy import deepcopy
from sklearn.svm import LinearSVR
import seaborn as sb




#regressor_test(data_complete,data_incomplete,years)

def regressor_test(complete,incomplete,years):
    kn_errors = []
    linear_errors = []
    svr_errors = []    
    
    for i in years[0]:
            
        X_train, X_test, y_train, y_test = train_test_split(complete.loc[:,complete.columns != i].values,
                                                            complete.loc[:,i].values, test_size = 0.2, random_state = 0)
        
        regressor1 = KNeighborsRegressor(2, 
                                       weights ='distance', 
                                       metric = 'euclidean')
        regressor2= LinearRegression()
        regressor3=LinearSVR()
        
        
        trained_model1 = regressor1.fit(X_train, 
                                 y_train)
        trained_model2 = regressor2.fit(X_train, 
                                 y_train)
        trained_model3 = regressor3.fit(X_train, 
                                 y_train)  
        
        incomplete_2 = deepcopy(incomplete)
        incomplete_2.loc[:, incomplete.columns != i] = incomplete_2.loc[:, 
                                incomplete.columns != i].apply(lambda row: row.fillna(row.mean()), axis=1)

        y_pred1 = regressor1.predict(X_test)
        y_pred2 = regressor2.predict(X_test)
        y_pred3 = regressor3.predict(X_test)
        
        
        kn_errors.append(mean_squared_error(y_test, y_pred1))
        linear_errors.append(mean_squared_error(y_test, y_pred2))
        svr_errors.append(mean_squared_error(y_test, y_pred3))
        
        
        #Test for checking the best model 
    MSE= []

    for i in range(0, len(complete.loc[:,'2007':'2017'].columns)):
        l = []
        l.extend((kn_errors[i], linear_errors[i], svr_errors[i]))
        
        if min(l) == kn_errors[i]:
            MSE.append("KNN")
        elif min(l) == linear_errors[i]:
            MSE.append("Linear")
        elif min(l) == svr_errors[i]:
            MSE.append("SVR")

    
    print("KNN =",MSE.count("KNN"),'\nLinear =',MSE.count("Linear") ,'\nSVR =',MSE.count("SVR"))


    return max(set(MSE), key = MSE.count)

#regress(choice,complete, incomplete,dataset,years)
    
def regress(choice,complete, incomplete,dataset,years):

    for i in years[0]:
        if choice == 'KNN':
            regressor = KNeighborsRegressor(2, 
                                            weights ='distance', 
                                            metric = 'euclidean')
        elif choice == 'SVR':
            regressor = LinearSVR()
        elif choice == 'Linear':
            regressor = LinearRegression()
            
        X_train, X_test, y_train, y_test = train_test_split(complete.loc[:,complete.columns != i].values,
                                                            complete.loc[:,i].values, test_size = 0.2, random_state = 0)
        
        trained_model = regressor.fit(X_train, 
                                 y_train)
        
        incomplete_2 = deepcopy(incomplete)
        incomplete_2.loc[:, incomplete.columns != i] = incomplete_2.loc[:, 
                                incomplete.columns != i].apply(lambda row: row.fillna(row.mean()), axis=1)
        prediction = trained_model.predict(incomplete_2.loc[:,incomplete_2.columns != i])
        temp_df = pd.DataFrame(prediction.reshape(-1,1), columns = [i])
        
        #now we are filling data_arrivals_incomplete 
        for index in range(len(temp_df)):
            if np.isnan(incomplete[i][index]):
                incomplete[i][index] = temp_df[i][index]



    #and filling the nan's on arrivals_df
    dataset.loc[:,'2007':'2017'] = pd.concat([complete, incomplete])
    
    
    return dataset


def download_data(data_url, metadata_url):
        
        #Read the dataset CSV
    data=pd.read_csv(data_url, header=2)
    
        #Select the columns with usefull data we need 2007 here to calculate the grow from 2007 to 2008
    data=data[['Country Name', 'Country Code','2007' , '2008', '2009', '2010', '2011',
                       '2012', '2013', '2014', '2015', '2016', '2017']]
        
        #Load metadata CSV
    metadata_country=pd.read_csv(metadata_url, header=0) 
    
        #Merge data CSV with Metadata CSV
    data_df=data.merge(metadata_country, on='Country Code', how='left') 
        
        #Set Country Name as Index
    new_index = data_df['Country Name']
    data_df.set_index(new_index,inplace=True )
    
        #Create a column named Is_Country for later removing the "areas" like asia
    data_df['Is_Country'] = data_df['Region'].notnull()
    
        #Drop unnecessary columns
    data_df.drop(['Country Name', 'Unnamed: 5', 'Region', 'IncomeGroup', 'SpecialNotes',
                  'TableName'], inplace=True, axis=1)
    
        #drop the 'areas'
    data_df = data_df[data_df.Is_Country != False]
    
        #drop the Is_Country column becouse we dont need it anymore
    data_df.drop('Is_Country', inplace=True, axis=1)
    
        #drop rows with 3 or more NANs values
    data_df.dropna(thresh=(len(data_df.loc[:,'2008':'2017'].columns) - 1), inplace=True, axis=0)
    

    return data_df



def common_fm(data_df):
        #create feature Avarage in Last 10 Years
    data_df['Avg_10_Years'] = data_df.loc[:,'2008':'2017'].mean(axis=1)
    
        #Create the feature Growth in 10 years
    data_df['Growth10ys']=(data_df['2017']/data_df['2008']-1)
    
    
    annual_growth = pd.DataFrame(index=data_df.index.values)
    
        #Fill the growth of each year in annual_arrival_growth dataframe
    for i in arrivals_df.loc[:,'2008':'2017'].columns:
        annual_growth[i] = (data_df[i] - data_df[str(int(i)-1)]) / data_df[str(int(i)-1)]
       
        #New Growth metric, becouse the last one wasnt good.
    data_df["AVG_Growth"] =  annual_growth.mean(axis=1)
    
        #sort by the best avarage arrivals in the last 10 years
    data_df = data_df.sort_values('Avg_10_Years', ascending =False)

        #divide each value of Growth in 10 years for the sum of the column
    data_df['% growth'] = data_df['AVG_Growth'] / data_df['AVG_Growth'].sum()

        #divide each value of Avarage in 10 years for the sum of the column
    data_df['%Avg'] = data_df['Avg_10_Years'] / data_df['Avg_10_Years'].sum()


    return data_df


#Define the URI from the github files (Number of Arrivels and Income)
income_url = 'https://raw.githubusercontent.com/rennanvoa2/Programming_DS_Project/master/Income.csv?token=AGBCKJVXIT3ASEMYSFAM2X25XA52W'
arrival_url = 'https://raw.githubusercontent.com/rennanvoa2/Programming_DS_Project/master/International%20Arrivals.csv?token=AGBCKJX7PXSB72QEPXFR37S5XA5WU'
metadata_url = 'https://raw.githubusercontent.com/rennanvoa2/Programming_DS_Project/master/Metadata_Country.csv?token=AGBCKJRD4VFGTAHO5I6I2BS5XA54Q'



arrivals_df = download_data(arrival_url,metadata_url)

#Count number of itens in the DataFrame
arrivals_df['Country Code'].count()

#Count not nulls in column 2007
arrivals_df['2007'].count()

#count nulls in 2007
arrivals_df['2007'].isnull().sum()

len(arrivals_df.columns)
#Check if the number of itens is correspondent after drop the rows
arrivals_df['Country Code'].count()



income_df=download_data(income_url,metadata_url)

#Count number of itens in the DataFrame
income_df['Country Code'].count()

#Count not nulls in column 2008
income_df['2008'].count()

#count nulls in 2008
income_df['2008'].isnull().sum()


########################################################################################
#                              REGRESSION TEST AND FIRST APPLICATION
########################################################################################


#preparing arrival data for regressors
data_arrivals = arrivals_df[['2007','2008','2009','2010','2011','2012','2013','2014', '2015','2016', '2017']]
data_arrivals_complete = pd.DataFrame()
data_arrivals_incomplete = data_arrivals[data_arrivals.isna().any(axis=1)]
data_arrivals_complete = data_arrivals[~data_arrivals.isna().any(axis=1)]

#dataframe with the name of the columns
years = pd.DataFrame(data_arrivals.columns)

#applying created functions
choice= regressor_test(data_arrivals_complete,data_arrivals_incomplete,years)
print('Best typ of regression to be used for arrivals prediction->',choice)
arrivals_df=regress(choice,data_arrivals_complete,data_arrivals_incomplete,arrivals_df,years)



#preparing income data for regressors
data_income = income_df[['2007','2008','2009','2010','2011','2012','2013','2014', '2015','2016', '2017']]
data_income_complete = pd.DataFrame()
data_income_incomplete = data_income[data_income.isna().any(axis=1)]
data_income_complete = data_income[~data_income.isna().any(axis=1)]

#dataframe with the name of the columns
years = pd.DataFrame(data_income.columns)

#applying created functions
choice= regressor_test(data_income_complete,data_income_incomplete,years)
print('Best type of regression to be used for income prediction ->',choice)
income_df=regress(choice, data_income_complete, data_income_incomplete, income_df, years)


########################################################################################
#                                CREATING FEATURES/METRICS
########################################################################################

#Weigh's for metrics
arrivals_total_number_weight = 1
arrivals_growth_weight = 1
income_total_number_weight = 1
income_growth_weight = 1
avg_per_person_weight = 1

###
#for arrivals
###

arrivals_df = common_fm(arrivals_df)

#Calculate the avarage between Growth and Avarage Numbers of Arrivals
arrivals_df['Growth x Average'] = (arrivals_growth_weight * arrivals_df['% growth'] +
               (arrivals_total_number_weight * arrivals_df['%Avg'])) / (arrivals_total_number_weight + arrivals_growth_weight)

#create a dataframe sorted by Growth X Avarage
Arrivals_in_growth_vs_arrivals = arrivals_df.sort_values('Growth x Average', ascending=False)


###
#for income
###

income_df = common_fm(income_df)

#Create Avarage expenditure per person
income_df['AVG_expenditure_per_person'] = income_df['Avg_10_Years'] / arrivals_df['Avg_10_Years']


#divide each value of Avarage per person for the sum of the column
income_df['%Avg_Per_Person'] = income_df['AVG_expenditure_per_person'] / income_df['AVG_expenditure_per_person'].sum()

#Calculate the avarage between Growth and Avarage Numbers of Arrivals
income_df['Growth x Average x Avg Exp'] = (income_growth_weight * income_df['% growth'] +
               (income_total_number_weight* income_df['%Avg']) + 
               avg_per_person_weight * income_df['%Avg_Per_Person']) / (income_total_number_weight + 
                                                income_growth_weight + avg_per_person_weight)



#create a dataframe sorted by Growth X Avarage
income_in_growth_vs_income = income_df.sort_values('Growth x Average x Avg Exp', ascending=False)


########################################################################################
#                                CLEANING/DROPPING PROBLEMATIC DATA
########################################################################################


#Drop Belarus, its an outlier in Arrivals Dataset
Arrivals_in_growth_vs_arrivals = Arrivals_in_growth_vs_arrivals.drop(['Belarus'])

#Drop Congo, its an outlier in income Dataset
income_in_growth_vs_income = income_in_growth_vs_income.drop(['Congo, Dem. Rep.'])

#________________________________________________________

#get the best 10 results in Arrivals
arrival_top_10 = Arrivals_in_growth_vs_arrivals.iloc[0:10,:]

#get the best 10 results in income
income_top_10 = income_in_growth_vs_income.iloc[0:10,:]


########################################################################################
#                                GRAPHING AND PLOTTING
########################################################################################


#New Graphs
#Arrivals

ax = sb.barplot(y= arrival_top_10['Growth x Average'], x = arrival_top_10.index.values, data = arrival_top_10, palette=("Blues_d"))
plt.ylabel("% Growth")
plt.title('Top 10 Arrivals')
plt.savefig('Arrivals.png')
sb.set_context("poster")
plt.show()

#Income

ax2 = sb.barplot(y= income_top_10['Growth x Average x Avg Exp'], x = income_top_10.index.values, data = income_top_10, palette=("Greens_d"))
plt.ylabel("% Growth")
plt.title('Top 10 Income')
plt.savefig('Income.png')
sb.set_context("poster")
plt.show()


#________________________________________________________

#Create a list with unique countries in arrival_top_10 + income_top_10
label_array = list(set(list(arrival_top_10.index.values)+ list(income_top_10.index.values)))


#Create a dataframe with the sum of top arrivals + top income
Arrivals_plus_income = pd.DataFrame(index=label_array)
Arrivals_plus_income['Arrivals GxA'] = arrivals_df['Growth x Average']
Arrivals_plus_income['Income GxA'] = income_df['Growth x Average x Avg Exp']

#sort the DataFrame by income
Arrivals_plus_income = Arrivals_plus_income.sort_values('Income GxA', ascending =False)


#Plot  income Values on a Bar chart
plt.bar(Arrivals_plus_income.index.values, Arrivals_plus_income['Income GxA'], width=0.5,
        label='Income', color='orange')

#Plot  Arrivals Values on a Bar chart
plt.bar(Arrivals_plus_income.index.values, Arrivals_plus_income['Arrivals GxA'], width=0.5,
        label='Arrival', bottom=Arrivals_plus_income['Income GxA'], color='blue')

#Resize the figure to 100x20
plt.rcParams['figure.figsize'] = (30,10)

# naming the x axis 
plt.xlabel('Top Countries') 

# naming the y axis 
plt.ylabel('Income & Arrivals (Growth x Average)') 
  
# giving a title
plt.title('Growth x Total Number of Arrivals') 

#Put legend
plt.legend()

#Save it in a png file for better view
plt.savefig('Arrivals x Income.png')

# function to show the plot 
plt.show()
