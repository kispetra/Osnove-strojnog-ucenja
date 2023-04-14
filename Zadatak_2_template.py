import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("test_6.jpg")

# prikazi originalnu sliku
plt.figure()
plt.subplot(1, 2, 1)
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w, h, d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()
km = KMeans(n_clusters=5, init="random", n_init=5, random_state=0)
km.fit(img_array_aprox)
labels = km.predict(img_array_aprox)

centers = km.cluster_centers_

new_img = centers[labels]
new_img = new_img.reshape(img.shape)

plt.subplot(1, 2, 2)
plt.title("Kvantizirana slika")
plt.imshow(new_img)
plt.tight_layout()
plt.show()
