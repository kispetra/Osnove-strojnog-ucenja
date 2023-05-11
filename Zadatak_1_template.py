import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

# generiranje podatkovnih primjera
X = generate_data(500, 3)
#broj grupa u podacima se lako moze prepoznati uz pomoc vizualizacije (dijagram rasprsenja) za svaki od nacina generiranja podataka (1-5)

# prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
plt.show()

#B
#Primijenite metodu K srednjih vrijednosti te ponovo prikažite primjere, ali svaki primjer
#obojite ovisno o njegovoj pripadnosti pojedinoj grupi. Nekoliko puta pokrenite programski
#kod. Mijenjate broj K. Što primjecujete?

kmeans = KMeans(n_clusters=4, init ='random')
kmeans.fit(X)
labels = kmeans.predict(X)
plt.figure()
plt.scatter(X[:,0],X[:,1], c=labels, cmap='viridis')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Grupirani podatkovni primjeri')
plt.show()
#neispravnim postavljanjem broja k dobija se previše ili premalo grupa
#kmeans kod nekih primjera ne grupira kako treba jer pretpostavlja da su grupe sferične, podjednake velicine i slicne gustoce,
#ne radi dobro s grupama nepravilnih oblika (jer radi na principu udaljenosti) (uz primjenu optimalnih vrijednosti k)
#kada flagc=1, radi dobro jer su grupe sfericne

#-------DRUGI NAČIN

for i in range(1, 6):
    #generiramo podatke
    x=generate_data(500, i)
    #inicializacija algoritma K srednjih vrijednosti
    #koliko imamo centara, nacin inicializacije centara(default k-means++), koliko puta ce se izvrsiti algoritam i random state
    km = KMeans(n_clusters=i, init ='random', n_init =5 , random_state =0)
    km.fit(x)
    labels = km.predict(x)

    # prikazi primjere u obliku dijagrama rasprsenja
    plt.figure()
    #na koliko podjela ide toliko boja imamo
    #prvi stupac, drugi stupac i treća kolona je boja
    #koliko prediktanih centara ima imati ce toliko razlicitih boja
    plt.scatter(x[:,0], x[:,1], c=labels)                        
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('podatkovni primjeri')
    plt.show()


#C
#Mijenjajte nacin definiranja umjetnih primjera te promatrajte rezultate grupiranja podataka
#(koristite optimalni broj grupa). Kako komentirate dobivene rezultate?