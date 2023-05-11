import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl


#Na temelju rješenja prethodnog zadatka izradite model koji koristi i kategoricku
#varijable „Fuel Type“ kao ulaznu velicinu. Pri tome koristite 1-od-K kodiranje kategorickih
#veliˇcina. Radi jednostavnosti nemojte skalirati ulazne velicine. Komentirajte dobivene rezultate.
#Kolika je maksimalna pogreška u procjeni emisije C02 plinova u g/km? O kojem se modelu
#vozila radi?

data=pd.read_csv('C:/Users/eetkisp/Desktop/osnove/lv4/data_C02_emission.csv')

data = data.drop(['Make', 'Model'], axis=1)                     

input = [
    'Engine Size (L)', 
    'Cylinders', 
    'Fuel Consumption City (L/100km)', 
    'Fuel Consumption Hwy (L/100km)', 
    'Fuel Consumption Comb (L/100km)', 
    'Fuel Consumption Hwy (L/100km)', 
    'Fuel Type'
]

output = ['CO2 Emissions (g/km)']

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder ()
X_encoded = ohe.fit_transform(data[['Fuel Type']]).toarray()
data['Fuel Type'] = X_encoded #zamjena sa enkodiranim

X = data[input].to_numpy()                                          #VAŽNO iz dataframe-a u ndarray pretvori UVIJEK DODAT PRIJE train_test_split
y = data[output].to_numpy()

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split (X , y , test_size = 0.2 , random_state =1 )

import sklearn.linear_model as lm
linearModel = lm.LinearRegression()
linearModel.fit( X_train, y_train )

print(linearModel.intercept_, linearModel.coef_)

y_test_p = linearModel.predict(X_test)          #uvijek prediktamo nakon što istreniramo model

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)
#prikaz samo jedne vel
plt.figure()
plt.scatter(x=X_test_n[:, 0], y=y_test, c="b",s=1)
plt.scatter(x=X_test_n[:, 0], y=y_test_p, c="r",s=1)
plt.xlabel('Engine Size (L)')
plt.ylabel('CO2 Emissions (g/km)')
plt.show()
#svih inputa u ovisnosti o y i SKALIRANE vel.
plt.figure()
for i in range(6):
    plt.scatter(X_test_n[:, i], y_test, c='blue', s=1)
    plt.scatter(X_test_n[:, i], y_test_p, c='red', s=1)
    plt.xlabel(input[i])
    plt.ylabel('CO2 Emissions (g/km)')
    plt.legend(('Real output', 'Predicted output'))
    plt.show()

from sklearn . metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import math
MAE=mean_absolute_error(y_test,y_test_p)
MSE= mean_squared_error(y_test,y_test_p)
RMSE= math.sqrt(MSE)
MAPE= mean_absolute_percentage_error(y_test,y_test_p)
R= r2_score(y_test,y_test_p)

print(f'MSE= {MSE}\nRMSE= {RMSE}\nMAE= {MAE}\nMAPE= {MAPE}\nR^2= {R}\n\n')







