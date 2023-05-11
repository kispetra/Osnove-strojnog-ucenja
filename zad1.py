import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sklearn.linear_model as lm #za lin regresiju
from sklearn.metrics import mean_absolute_error #za procjenu
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

data=pd.read_csv('C:/Users/eetkisp/Desktop/osnove/lv4/data_C02_emission.csv')

#A 
#Odaberite željene numericke velicine specificiranjem liste s nazivima stupaca. Podijelite
#podatke na skup za ucenje i skup za testiranje u omjeru 80%-20%.

input_variables = [
    "Fuel Consumption City (L/100km)",
    "Fuel Consumption Hwy (L/100km)",
    "Fuel Consumption Comb (L/100km)",
    "Fuel Consumption Comb (mpg)",
    "Engine Size (L)",
    "Cylinders",
] #ostale numericke ulazne vel

output_variable = ['CO2 Emissions (g/km)'] 

X=data[input_variables].to_numpy()
y=data[output_variable].to_numpy()

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.2, random_state=1 )


#B
#Pomocu matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova
#o jednoj numerickoj velicini.

plt.figure()
plt.scatter(X_train[:,0], y_train, c='blue', s=1)
plt.scatter(X_test[:,0], y_test, c='red', s=1) #ako promjenim broj samo se doda vise rezultata
plt.xlabel('Fuel Consumption Comb (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
#plt.show()



#C
#Izvršite standardizaciju ulaznih velicina skupa za ucenje. Prikažite histogram vrijednosti
#jedne ulazne velicine prije i nakon skaliranja. Na temelju dobivenih parametara skaliranja
#transformirajte ulazne velicine skupa podataka za testiranje.

plt.figure()
plt.hist(X_train[:,4])#HISTOGRAM PRIJE SKALIRANJA

sc = MinMaxScaler()
X_train_n = sc.fit_transform ( X_train )
X_test_n = sc.transform ( X_test )

plt.figure()
plt.hist(X_train_n[:,4]) #POSLIJE SKALIRANJA
plt.show()

#-- drugi nacin
plt.figure()
plt.hist(X_train[::,2], color='blue')
plt.hist(X_train_n[::,2], color='red')
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('Frequency')
plt.legend(('Before Scaling','After Scaling'))
#plt.show()

sc = MinMaxScaler()                                             #SKALIRANJE VELICINA
X_train_n = sc.fit_transform( X_train )                         #namjesti veličine i skalira da bi bilo izmedu 0 i 1 
X_test_n = sc.transform( X_test )                               #transformira veličine tj. skalira i testne podatke kako bi bili kompatibil i sa onima iz train skupa

for i in range(len(input_variables)): #za svih 6 ulaznih 
    
    ax1 = plt.subplot(211)                                      #2 reda, 1 stupac i 1 index
    ax1.hist(x=X_train[:, i])                   
    ax2 = plt.subplot(212)                                      #2 reda, 1 stupca i 2 index
    ax2.hist(x=X_train_n[:, i])
    plt.show()

#D
#Izgradite linearni regresijski modeli. Ispišite u terminal dobivene parametre modela i
#povežite ih s izrazom 4.6.
linearModel = lm.LinearRegression()
linearModel.fit( X_train_n , y_train )

print(linearModel.intercept_, linearModel.coef_)

#E
#Izvršite procjenu izlazne velicine na temelju ulaznih velicina skupa za testiranje. Prikažite
#pomocu dijagrama raspršenja odnos izmedu stvarnih vrijednosti izlazne velicine i procjene
#dobivene modelom.
y_test_p = linearModel.predict(X_test_n)

plt.figure()
plt.scatter(x=X_test_n[:, 0], y=y_test, c="b")
plt.scatter(x=X_test_n[:, 0], y=y_test_p, c="r")
plt.show()

#F
#Izvršite vrednovanje modela na naˇcin da izraˇcunate vrijednosti regresijskih metrika na
#skupu podataka za testiranje.
MAE = mean_absolute_error(y_test, y_test_p)                     #ERRORI!!!!!!!!!!!!!!!!
RMSE = mean_squared_error(y_test, y_test_p, squared=False)      #VRSTE FUNKCIJA ZA IZRAČUN ERORA
MAPE = mean_absolute_percentage_error(y_test, y_test_p)
R2 = r2_score(y_test, y_test_p)

print("MAE", MAE)
print("RMSE", RMSE)
print("MAPE", MAPE)
print("R2", R2)








