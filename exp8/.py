import tensorflow as tf

from tensorflow.keras import layers, models
 
import matplotlib.pyplot as plt import numpy as np


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()



train_images = train_images / 255.0 test_images = test_images / 255.0


train_images = train_images.reshape((60000, 28, 28, 1))

test_images = test_images.reshape((10000, 28, 28, 1))



model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1))) model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu')) model.add(layers.MaxPooling2D((2,2))) model.add(layers.Flatten()) model.add(layers.Dense(64, activation='relu')) model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
 
metrics=['accuracy'])



history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))


test_loss, test_acc = model.evaluate(test_images, test_labels) print("Test Accuracy:", test_acc)


plt.plot(history.history['accuracy'], label='Train Accuracy') plt.plot(history.history['val_accuracy'], label='Validation Accuracy') plt.legend()
plt.show()



class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat', 'Sandal','Shirt','Sneaker','Bag','Ankle boot']


plt.figure(figsize=(10,5)) for i in range(10):
plt.subplot(2,5,i+1) plt.imshow(train_images[i].reshape(28,28), cmap='gray') plt.title(class_names[train_labels[i]])
plt.axis('off') plt.show()
