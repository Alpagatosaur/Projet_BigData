from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten , Conv2D, MaxPooling2D

data = mnist.load_data()
train, test = data
X_train, y_train = train

X_train = X_train/ 255

X_train = X_train.reshape(60000, 28, 28, 1)

model_conv = Sequential()
model_conv.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
model_conv.add(MaxPooling2D(2,2))
#dropout
model_conv.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))


model_conv.add(Flatten())
#partie 2 

model_conv.add(Dense(300, activation="relu"))
model_conv.add(Dense(250, activation="relu"))
model_conv.add(Dense(10, activation="softmax"))


model_conv.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

model_conv.fit(X_train, y_train, epochs=1)
model_conv.save('test.h5')
