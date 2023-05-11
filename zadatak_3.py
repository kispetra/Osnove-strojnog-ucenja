
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


model = load_model('Model/')

img = Image.open('C:/Users/eetkisp/Desktop/osnove/lv8/test1.png').convert("L")
img = img.resize((28, 28))

img_array = np.array(img)
img_array = img_array.astype("float32") / 255
img_array = np.expand_dims(img_array, -1)
img_array = np.reshape(img_array, (1, 784))

plt.figure()
plt.imshow(img)
plt.show()
prediction = model.predict(img_array)
pred = np.argmax(prediction, axis=1)
print(pred)
