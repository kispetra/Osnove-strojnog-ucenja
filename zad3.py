import numpy as np
import matplotlib.pyplot as plt

slika=plt.imread("C:/Users/eetkisp/Desktop/osnove/lv2/road.jpg")
print(slika.shape) #koordinate ispise
#opceniti prikaz slike
plt.figure()
plt.imshow(slika)
plt.show()

#posvjetli sliku
plt.figure()
plt.imshow(slika, alpha=0.8)
plt.show()

#druga cetvrtina slike po sirini
rows, cols, pixels = slika.shape
imgc = slika[:,round(cols/4):round(cols/2),:].copy()
plt.figure()
plt.imshow(imgc)
plt.show()

#rotacija 90
imgr = np.rot90(slika, axes=(1,0)) #govori axes u koju stranu
plt.figure()
plt.imshow(imgr)
plt.show()

#zrcaliti
imgf = np.flip(slika, axis=1)#za to zadužena funkcija flip, kada je 1 rotira se prema y, a 0 po x
plt.figure()
plt.imshow(imgf)
plt.show()




