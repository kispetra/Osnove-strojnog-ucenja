import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn . preprocessing import MinMaxScaler
import sklearn . linear_model as lm

data = pd.read_csv('data_C02_emission.csv')
data = data.drop(["Make","Model"], axis=1)

#A
input_var = ['Fuel Consumption City (L/100km)',
        'Fuel Consumption Hwy (L/100km)',
        'Fuel Consumption Comb (L/100km)',
        'Fuel Consumption Comb (mpg)',
        'Engine Size (L)',
        'Cylinders']

output_var=['CO2 Emissions (g/km)']

X=data[input_var].to_numpy()
y=data[output_var].to_numpy()

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.2, random_state=1 )
#B
fig=plt.figure(figsize=(8,6))
for i in range(0,6):
    plt.scatter(X_train[:, i], y_train, c="blue", label="Training data", s=1)
    plt.scatter(X_test[:, i], y_test, c="red", label="Test data", s=1)
    #plt.show()

#C
sc=MinMaxScaler()
X_train_n=sc.fit_transform(X_train)
X_test_n=sc.transform(X_test)

for i in range(0,6):
    fig, axs=plt.subplots(2)
    axs[0].hist(X_train[:,i], bins=30, color='blue')
    axs[0].set_title('Normal')
    axs[1].hist(X_train_n[:, i], bins=30, color='red')
    axs[1].set_title('Standardized')
    #plt.show()

#D
linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)
print(linearModel.get_params())

