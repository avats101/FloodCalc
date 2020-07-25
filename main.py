import pandas as pd # Importing the libraries
dataset = pd.read_csv('Data.csv')  # Importing the dataset
rain = dataset.iloc[:, 10:13].values #Values of Oct,Nov,Dec Rainfall
result = dataset.iloc[:, 23:26].values #Values of Oct,Nov,Dec flood Status

#Applying Feature Scaling to rain,result
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
rain = sc.fit_transform(rain)
result = sc.transform(result)

#Using Random Forest, as the number of 1's in the DataSet is less so to deal with it using 1000 decision trees
from sklearn.ensemble import RandomForestRegressor 
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(rain,result)

#Taking Input from user regarding the Amount of rainfall in month to be checked,also taking the last 2 months rainfall too.
current_month=int(input("Amount of rainfall in month to be checked: "))
month_1=int(input("Amount of rainfall in the previous month to this given current_month: "))
month_2=int(input("Amount of rainfall in the previous month to this given month_1: "))


answer=regressor.predict(sc.transform([[current_month,month_1,month_2]])) 
answer= sc.inverse_transform(answer)
percent=(answer[0][2])*100
print(round(percent,2),"% chance that flood will occur for the given month.")

