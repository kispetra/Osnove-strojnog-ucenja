import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

#A
#Prikažite podatke za ucenje u x1−x2 ravnini matplotlib biblioteke pri cemu podatke obojite
#s obzirom na klasu. Prikažite i podatke iz skupa za testiranje, ali za njih koristite drugi
#marker (npr. ’x’). Koristite funkciju scatter koja osim podataka prima i parametre c i
#cmap kojima je moguce definirati boju svake klase.

plt.figure()
plt.scatter(x=X_train[:, 0], y=X_train[:, 1], c=y_train, cmap="coolwarm")  #coolwarm je raspon boja crvena-plava
plt.scatter(x=X_test[:, 0], y=X_test[:, 1], c=y_test, cmap="coolwarm", marker="x")
plt.show()

#B
#Izgradite model logisticke regresije pomocu scikit-learn biblioteke na temelju skupa podataka
#za ucenje.

from sklearn.linear_model import LogisticRegression
LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

print("Koeficijenti:", LogRegression_model.coef_)
print("Presječna točka: ", LogRegression_model.intercept_)                                                 #THETA0

#C
#Pronadite u atributima izgradenog modela parametre modela. Prikažite granicu odluke
#naucenog modela u ravnini x1 −x2 zajedno s podacima za ucenje. Napomena: granica
#odluke u ravnini x1−x2 definirana je kao krivulja: θ0+θ1x1+θ2x2 = 0.
from sklearn . linear_model import LogisticRegression

coef = LogRegression_model.coef_[0]
intercept = LogRegression_model.intercept_

# Define the decision boundary as a function of x1
def decision_boundary(x1):
    return (-coef[0]*x1 - intercept) / coef[1]
# Plot the decision boundary along with the training data
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm')
plt.plot(X_train[:, 0], decision_boundary(X_train[:, 0]))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Logistic Regression Decision Boundary')
plt.show()

#D
#Provedite klasifikaciju skupa podataka za testiranje pomocu izgradenog modela logisticke
#regresije. Izracunajte i prikažite matricu zabune na testnim podacima. Izracunate tocnost,
#reciznost i odziv na skupu podataka za testiranje.
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay

y_pred = LogRegression_model.predict(X_test) #MORAMO PREDICTAT TEST SKUP DA VIDIMO KAK RADI zato je X_test
cm = confusion_matrix(y_test, y_pred)
print("Matrica zabune: ", cm)
disp = ConfusionMatrixDisplay (confusion_matrix ( y_test , y_pred ) )
disp.plot ()
plt.show ()

from sklearn.metrics import accuracy_score, precision_score, recall_score
print (" Tocnost : " , accuracy_score ( y_test , y_pred ) )
print(" Preciznost : ", precision_score (y_test, y_pred))
print(" Odziv: ", recall_score(y_test, y_pred))

#E
#Prikažite skup za testiranje u ravnini x1−x2. Zelenom bojom oznacite dobro klasificirane
#primjere dok pogrešno klasificirane primjere oznacite crnom bojom.
plt.figure()
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        color='green'
    else:
        color='black'

    plt.scatter(X_test[i,0],X_test[i,1],c=color)
plt.show()
#---DRUGI NACIN
correctly_classified = y_test ==  y_pred
plt.scatter(
    X_test[correctly_classified, 0],       #LIJEVO redovi DESNO stupci, uzela 1. supac
    X_test[correctly_classified, 1],       #uzela 2. stupac i prikatala ih kao koordinate u scatterplot-u
    c = "green",
    marker = "o"
)

incorrectly_classified = y_test != y_pred
plt.scatter(
    X_test[incorrectly_classified, 0],
    X_test[incorrectly_classified, 1],
    c="black",
    marker="x"
)

plt.show()




