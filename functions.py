#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:26:27 2019

@author: hvjanuario
"""

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
    kn_errors = []
    linear_errors = []
    svr_errors = []    
    
    
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


#to be used instead of the FOR cycles for arrivals

choice= regressor_test(data_arrivals_complete,data_arrivals_incomplete,years)
print('Best typ of regression to be used ->',choice)
arrivals_df=regress(choice,data_arrivals_complete,data_arrivals_incomplete,arrivals_df,years)



#to be used instead of the FOR cycles for income

choice= regressor_test(data_income_complete,data_income_incomplete,years)
print('Best typ of regression to be used ->',choice)
income_df=regress(choice,data_income_complete,data_income_incomplete,income_df,years)






