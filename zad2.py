import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Users/eetkisp/Desktop/osnove/lv3/data_C02_emission.csv")

#A
#Pomocu histograma prikažite emisiju C02 plinova. Komentirajte dobiveni prikaz
data['CO2 Emissions (g/km)'].plot(kind='hist', bins=20)
plt.xlabel('CO2 Emission (g/km)')
plt.show()

#B
#Pomo´cu dijagrama raspršenja prikažite odnos izme ¯ du gradske potrošnje goriva i emisije C02 plinova
data["Fuel Color"] = data["Fuel Type"].map(
    {
        "X": "Red",
        "Z": "Blue",
        "D": "Green",
        "E": "Purple",
        "N": "Yellow",
    }
)
data.plot.scatter(
    x="Fuel Consumption City (L/100km)",
    y="CO2 Emissions (g/km)",
    c="Fuel Color",
)
plt.show()

#C
#Pomocu kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s obzirom na tip goriva.
data.boxplot(column=['Fuel Consumption Hwy (L/100km)'], by='Fuel Type')
plt.show()

#D
#Pomocu stupcastog dijagrama prikažite broj vozila po tipu goriva. Koristite metodu groupby
grouped = data.groupby('Fuel Type').size()
grouped.plot(kind='bar')
plt.show()


#E
#Pomocu stupcastog grafa prikažite na istoj slici prosjecnu C02 emisiju vozila s obzirom na broj cilindara.
data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean().plot(kind="bar")
plt.show()