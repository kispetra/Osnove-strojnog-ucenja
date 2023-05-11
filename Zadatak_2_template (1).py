import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("C:/Users/eetkisp/Desktop/osnove/lv7/imgs/imgs/test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

print(f'Broj boja u originalnoj slici: {len(np.unique(img_array_aprox, axis=0))}') #ili .shape[0] za broj redova
print("Number of unique colors:", len(np.unique(img_array_aprox)))

#trazenje lakta, optimalnog broja grupa (K), lakat je uočljiv
squareSums = []
"""for i in range (1,10):
    kmeans = KMeans(n_clusters=i, init='random')
    kmeans.fit(img_array_aprox)
    squareSums.append(kmeans.inertia_)

plt.plot(range(1,10), squareSums)
plt.xlabel('K')
plt.show()
"""

#B
#Primijenite algoritam K srednjih vrijednosti koji ce pronaci grupe u RGB vrijednostima
#elemenata originalne slike.

km = KMeans(n_clusters=4, init='random', n_init=5, random_state=0)
km.fit(img_array_aprox)
labels = km.predict(img_array_aprox)

#C
#Vrijednost svakog elementa slike originalne slike zamijeni s njemu pripadaju´cim centrom
for i in range(len(labels)):
    img_array_aprox[i]=km.cluster_centers_[labels[i]]    #promjena svakog retka(boje) u jednu od k boja (najblizi i odgovarajuci centroid) - kvantizacija slike

print(f'Broj boja u aproksimiranoj slici: {len(np.unique(img_array_aprox, axis=0))} (jednak predodredenom broju grupa)')


#D
#Usporedite dobivenu sliku s originalnom. Mijenjate broj grupa K. Komentirajte dobivene
#rezultate.
img_aprox = np.reshape(img_array_aprox, (h,w,d))    #povratak na originalnu dimenziju slike 
img_aprox = (img_aprox*255).astype(np.uint8)        #povratak iz raspona 0 do 1 u int
plt.figure()
plt.title("Aproksimirana slika")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()


#E
#za sve dostupne slike
labels_unique = np.unique(labels)
for i in range(len(labels_unique)):
    binary_image = labels==labels_unique[i] #labels je n_pixela x 1 shape
    binary_image = np.reshape(binary_image, (h,w)) #potrebno reshapeat za prikaz nazad u normalne dimenzije slike(bez rgb dimenzije)
    plt.figure()
    plt.title(f"Binarna slika {i+1}. grupe boja")
    plt.imshow(binary_image)
    plt.tight_layout()
    plt.show()
#prikazom binarnih slika svake grupe, primjecuje se da su grupe disjunktni skupovi, tj. svaka grupa predstavlja jednu boju na slici

