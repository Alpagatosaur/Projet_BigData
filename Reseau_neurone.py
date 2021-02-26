## Neurones Denses
from tensorflow.keras.datasets import mnist

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

data = mnist.load_data()

train, test = data
X_train, y_train = train

X_train = X_train.reshape(60000, 28*28) / 255

model = Sequential() # crée moi un réseau de neurones vides

model.add(Dense(200, activation="relu", input_shape=(28**2,))) #ajoute moi une couche de neurones (avec 100 neurones)
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="softmax"))

sgd_optimizer = SGD(lr=3)

model.compile(loss="sparse_categorical_crossentropy", optimizer=sgd_optimizer, metrics=['accuracy'])


model.fit(X_train, y_train, epochs=5, batch_size=32)
model.evaluate(X_train, y_train)


##Neurones Convolutionels
from tensorflow.keras.datasets import mnist
data = mnist.load_data()
train, test = data
X_train, y_train = train
X_train = X_train/ 255

X_train = X_train.reshape(60000, 28, 28, 1)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, Dense

model_conv = Sequential()
model_conv.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
model_conv.add(MaxPooling2D(pool_size = (3,3)))
model_conv.add(Dropout(0.3)) #dropout
model_conv.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model_conv.add(MaxPooling2D(pool_size = (3,3)))

model_conv.add(Flatten())
#partie 2

model_conv.add(Dense(300, activation="relu"))
model_conv.add(Dense(250, activation="relu"))
model_conv.add(Dense(10, activation="softmax"))


model_conv.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

model_conv.fit(X_train, y_train, epochs=1)## 0.9120

## script qui charge le modèle, charge une image et prédit la classe de l'imag
from tensorflow.keras.datasets import mnist

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import cv2
from tensorflow import keras


#Charge le modele
data = mnist.load_data()
train, test = data
X_train, y_train = train
X_train = X_train.reshape(60000, 28*28) / 255
model = Sequential() # crée moi un réseau de neurones vides
model.add(Dense(10, activation="softmax"))
sgd_optimizer = SGD(lr=3)
model.compile(loss="sparse_categorical_crossentropy", optimizer=sgd_optimizer, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32)

model.save('D:/Users/tanch/Documents/VisualS/mnist_trained.h5')

#prediction
model = keras.models.load_model('D:/Users/tanch/Documents/VisualS/mnist_trained.h5')
img = cv2.imread('D:/Users/tanch/Documents/VisualS/BigData/example_mnist.png')
print(img.shape)

img=cv2.resize(img,(28,28))

img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img = img.reshape(1,28*28)
prediction = model.predict(img)

print(prediction)