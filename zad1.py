import pandas as pd
import numpy as np

data=pd.read_csv("C:/Users/eetkisp/Desktop/osnove/lv3/data_C02_emission.csv")

#A
print("DataFrame sadrži ", len(data)," mjerenja.")
print("Svaka veličina je tipa: ", data.dtypes) #moze i data.info()
print("Duplicirane vrijednosti: ", data.duplicated().sum())
print("Izostale vrijednosti po stupcima: ", data.isnull().sum())
#ako postoje brišemo 
data.drop_duplicates()
#nullvrj
data.dropna(axis=0) #redove
data.dropna(axis=1) #stupce
data=data.reset_index()#da nemamo rupe u indexima
#kategoricke konvertiraj u tip category
for col in data:
    if type(col) == object:
        data[col] = data[col].astype("Category")

#B
#3 automobila s najvecom i najmanjom potrosnjom, u terminal ime proivodaca, model i kolika je gradska potrosnja
sorted=data.sort_values(by="Fuel Consumption City (L/100km)")
print('Najmanja 3:')
print(sorted[['Make','Model','Fuel Consumption City (L/100km)']].head(3))
print('Najveća 3:')
print(sorted[['Make','Model','Fuel Consumption City (L/100km)']].tail(3))

#C
#koliko vozila ima prosjecnu velicinu izmedu 2.5L i 3.5L? Kolika je prosjeˇcna C02 emisija plinova za ova vozila?
print("Broj vozila prosjecne velicine motora izmedu: ", data[(data['Engine Size (L)'] >= 2.5 ) & (data['Engine Size (L)'] <= 3.5 )].Model.count())
new_data = data[(data['Engine Size (L)'] >= 2.5 ) & (data['Engine Size (L)'] <= 3.5 )]['CO2 Emissions (g/km)'].mean()
print("Prosječna emisija CO2: ", new_data)
#----- drugi nacin
engineRestrictedData = data[(data['Engine Size (L)'] >=2.5) & data["Engine Size (L)"]<= 3.5]
print(len(engineRestrictedData))
print("PROSJECNO CO2")
print(engineRestrictedData['CO2 Emissions (g/km)'].mean())

#D
#Koliko mjerenja se odnosi na vozila proizvodca Audi? Kolika je prosjecna emisija C02 plinova automobila proizvoaca Audi koji imaju 4 cilindara?
audi= data[(data['Make'] == 'Audi')]
print(audi.Model.count())
print(audi[(audi['Cylinders'] == 4)]['CO2 Emissions (g/km)'].mean()) 
#-----drugi nacin 
count_audi = data[(data['Make'] == 'Audi')]
print('Broj Audia:', len(count_audi))
emisija = data[(data['Cylinders'] == 4) & (data['Make']=='Audi')]['CO2 Emissions (g/km)'].mean()
print(emisija)

#E
#Koliko je vozila s 4,6,8...cilindara? Kolika je prosjeˇcna emisija C02 plinova s obzirom nabroj cilindara?
parni_cilindri=data[(data['Cylinders'])%2==0]
print(len(parni_cilindri))
print(parni_cilindri['CO2 Emissions (g/km)'].mean())

#F
#Kolika je prosječna gradska potrošnja u slučaju vozila koja koriste dizel, a kolika za vozila koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?
dizel=data[(data['Fuel Type']=='D')]
benzin=data[(data['Fuel Type']=='X') | (data['Fuel Type']=='Z')]
print(dizel['Fuel Consumption City (L/100km)'].mean())
print(benzin['Fuel Consumption City (L/100km)'].mean())
print(dizel['Fuel Consumption City (L/100km)'].median())
print(benzin['Fuel Consumption City (L/100km)'].median())

#G
#Koje vozilo s 4 cilindra koje koristi dizelski motor ima najvecu gradsku potrošnju goriva?
target = data[(data['Cylinders'] == 4) & (data['Fuel Type'] == 'D')].sort_values('Fuel Consumption City (L/100km)', ascending=False)
print(target.head(1)[['Make','Model','Fuel Consumption City (L/100km)']]) #da ne ispise sve

#H
#Koliko ima vozila ima ruˇcni tip mjenjaˇca (bez obzira na broj brzina)?
target2=data[(data['Transmission'].str.contains("M",case = False))]
manual = target2.reset_index(drop = True)
print(len(manual))
#----drugi nacin
manual_transmission = data[(data["Transmission"].str.startswith('M'))]
print("Postoji", manual_transmission.shape[0], "vozila s ručnim mjenjačem.")

#I
#Izracˇunajte korelaciju izmed¯u numericˇkih velicˇina. Komentirajte dobiveni rezultat
print(data.corr(numeric_only=True))


