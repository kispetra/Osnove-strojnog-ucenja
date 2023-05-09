import numpy as np
import matplotlib . pyplot as plt


file = np.genfromtxt("data.csv", delimiter=",", skip_header=1)

#A
print(len(file)) 

#B odnos visine i težine
height=file[:,1]
weight=file[:,2]
plt.title("Odnos visine i mase")
plt.xlabel("height")
plt.ylabel("weight")
plt.scatter(height , weight , marker ="P", s =1 )
plt.show()

#C za svaku 50 osobu
height_every50=file[::50,1]
weight_every50=file[::50,2]
plt.title("Odnos visine i mase za svakih 50")
plt.xlabel("height")
plt.ylabel("weight")
plt.scatter(height_every50 , weight_every50 , marker ="P", s =1 )
plt.show()

#D
print("Minimalna visina:", min(height))
print("Maksimalna visina:", max(height))
print("Srednja visina:", sum(height)/len(height))

#E
male=(file[:, 0]==1)
female=(file[:, 0]==0)

print("Minimalna visina za muške: ", file[male,1].min())
print("Maksimalna visina za muške: ", file[male,1].max())
print("Srednja visina za muške: ", file[male,1].mean())

print("Minimalna visina za muške: ", file[female,1].min())
print("Maksimalna visina za muške: ", file[female,1].max())
print("Srednja visina za muške: ", file[female,1].mean())

