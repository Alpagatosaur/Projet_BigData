import cv2
from tensorflow import

model = keras.models.load_model('test.h5')
img = cv2.imread('bug_cat.jpg')
print(img.shape)

img = cv2.resize(img,(28,28))
print(img.shape)

img = cv2.cvtColor(img, cv2.cv2COLOR_RGB2GRAY)
print(img.shape)

img = img.reshape(1,28,28,1)
prediction = model.predict(img)
print(prediction)
model.summary()
