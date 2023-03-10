
import numpy as np
import matplotlib . pyplot as plt

block_white = np.zeros((50,50))
block_black = 255 * np.ones((50,50))

column1=np.vstack((block_black,block_white))
column2=np.vstack((block_white,block_black))

final=np.hstack((column1,column2))

plt.figure()
plt.imshow(final, cmap="gray")
plt.show()
print ("a")
