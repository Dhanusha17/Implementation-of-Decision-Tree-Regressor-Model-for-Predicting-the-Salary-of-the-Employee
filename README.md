# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.
 

## Program:

/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by:DHANUSHA K

RegisterNumber:212223040034  
*/
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```
## Output:
## Initial dataset:

![238249522-3013f142-8d1c-42cc-9f11-751ce674ae84](https://github.com/Dhanusha17/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/151549957/1dd30655-380c-4d9a-8563-c66aa142a7e1)

## Data Info:

![238249556-f42b2f9a-d161-4205-90ca-368733c1c156](https://github.com/Dhanusha17/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/151549957/6f33db0b-bde0-430d-890a-a8f09c41e7de)

## Optimization of null values:

![238249588-d0eefc95-353f-4c35-b696-de3a978d8124](https://github.com/Dhanusha17/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/151549957/f97f5ad1-5702-45cd-9550-4672b7986a45)

## Converting string literals to numerical values using label encoder:

![238249653-4d7e562c-9a2a-48ca-ae0e-5b5f1a429e91](https://github.com/Dhanusha17/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/151549957/3c4dda0b-730c-473f-b426-9465bb1d5bc7)


## Assigning x and y values:

![238249701-c56e3af8-065e-45de-93a5-ce9910f6638d](https://github.com/Dhanusha17/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/151549957/1e37aa40-f052-4ede-ae9d-380d08993769)

## Mean Squared Error:

![238249743-29e082dc-8dc0-4836-ba51-334497170c7a](https://github.com/Dhanusha17/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/151549957/d49deae2-8c03-43b9-98cf-af9f565f767f)

## R2 (variance):

![238249774-0fe196e4-baea-44e6-8a90-240185d0cbe7](https://github.com/Dhanusha17/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/151549957/41fbf68f-1496-470a-b52a-fc12fd1b98dc)

## Prediction:

![238249776-e69d2f5c-67f7-462a-9ddc-20a51c383016](https://github.com/Dhanusha17/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/151549957/d25a1841-97fd-4bae-b48b-634aef652fd3)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
