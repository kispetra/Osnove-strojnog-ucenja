import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (X_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa
#Pomocu matplotlib biblioteke prikažite jednu sliku iz skupa podataka za ucenje te ispišite
#njezinu oznaku u terminal.
X_train_reshaped = np.reshape(X_train,(len(X_train),X_train.shape[1]*X_train.shape[2])) #umjesto len(X_train) moze i X_train.shape[0]
X_test_reshaped = np.reshape(X_test,(len(X_test),X_test.shape[1]*X_test.shape[2]))      #umjesto len(X_test) moze i X_test.shape[0]
plt.imshow(X_train[7])   #slike se prikazuju normalnim 2d poljem
plt.title(f'Slika broja {y_train[7]}')
plt.show()
print("Oznaka slike:", y_train[7])

#-----DRUGI NACIN-JEDNOSTAVNIJE
plt.imshow(X_train[4])
plt.show()
print("Oznaka slike:", y_train[4])


# skaliranje slike na raspon [0,1]
x_train_s = X_train.astype("float32") / 255
x_test_s = X_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)


#1.ZAD.Koliko primjera sadrži skup za ucenje, a koliko skup za
#testiranje?
print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")
#-----drugi nacin, ovo gore ona napisala
print(f'Broj primjera za ucenje: {len(X_train)}')
print(f'Broj primjera za testiranje: {len(X_test)}')
#ulazni podaci imaju oblik (broj primjera,28,28)(svaka slika je 28x28 piksela), svaki piksel predstavljen brojem 0-255
#izlazna velicina kodirana na nacin da su znamenke predstavljene brojevima 0-9
#svaka slika(primjer)-2d matrica, 28x28


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)


# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
model = keras.Sequential()
model.add(layers.Input(shape=(784,))) #ulaz 784 elementa
model.add(layers.Dense(units=100, activation="relu")) #1. skriveni sloj 100 neurona "relu"
model.add(layers.Dense(units=50, activation="relu")) #2. skriveni sloj, 50 neurona, "relu"
model.add(layers.Dense(units=10, activation="softmax")) #Izlazni sloj 10 neurona, softmax
#pomoću metode summary ispišite informacije o mreži u terminal
model.summary()

#oneHotEncoding izlaza, da sve bude prema skici u predlosku, za ovo u kerasu postoji isto funkcija y_train = keras.utils.to_categorical(y_train, num_classes=10)
from sklearn.preprocessing import OneHotEncoder
oh=OneHotEncoder()
y_train_encoded = oh.fit_transform(np.reshape(y_train,(-1,1))).toarray() #OneHotEncoder trazi 2d array, pa treba reshape (-1,1), tj (n,1),
y_test_encoded = oh.transform(np.reshape(y_test,(-1,1))).toarray() #-1 znaci sam skontaj koliko, mora toarray() obavezno kod onehotencodera

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=[
        "accuracy",
    ],
)

# TODO: provedi ucenje mreze
history = model.fit(X_train_reshaped , y_train_encoded, batch_size=32, epochs=2, validation_split=0.1)

#TODO: Izvršite EVALUACIJU mreže na testnom skupu podataka pomoću metode .evaluate
#evaluacija i ispis 
score = model.evaluate(X_test_reshaped, y_test_encoded, verbose=0)
for i in range(len(model.metrics_names)):
    print(f'{model.metrics_names[i]} = {score[i]}')


# TODO: Prikazi test accuracy i matricu zabune
y_predictions = model.predict(X_test_reshaped)  #vraca za svaki primjer vektor vjerojatnosti pripadanja svakoj od 10 klasa (softmax) (10 000,10)
y_predictions = np.argmax(y_predictions, axis=1)  #vraća polje indeksa najvecih elemenata u svakom pojedinom retku (1d polju) (0-9) (10 000,) - 1d polje
cm = confusion_matrix(y_test, y_predictions)    #zbog prethodnog koraka, usporedba s y_test, a ne encoded
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# TODO: spremi model
model.save('Model/')

