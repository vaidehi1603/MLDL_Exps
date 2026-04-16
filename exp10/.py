# Step 1: Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist

# Step 2: Load Dataset (MNIST)
(x_train, _), (x_test, _) = mnist.load_data()

# Step 3: Preprocess Data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = np.reshape(x_train, (-1, 28, 28, 1))
x_test = np.reshape(x_test, (-1, 28, 28, 1))

# Step 4: Add Noise
noise_factor = 0.5

x_train_noisy = x_train + noise_factor * np.random.normal(size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Step 5: Build Autoencoder Model

input_img = Input(shape=(28, 28, 1))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(encoded)
x = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

# Step 6: Compile Model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Step 7: Train Model
autoencoder.fit(
    x_train_noisy, x_train,
    epochs=10,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test_noisy, x_test)
)

# Step 8: Test / Predict
decoded_imgs = autoencoder.predict(x_test_noisy)

# Step 9: Display Results
n = 5
for i in range(n):
    plt.figure(figsize=(10, 4))

    # Noisy Image
    plt.subplot(1, 3, 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
    plt.title("Noisy")

    # Original Image
    plt.subplot(1, 3, 2)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")

    # Denoised Image
    plt.subplot(1, 3, 3)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Denoised")

    plt.show()
