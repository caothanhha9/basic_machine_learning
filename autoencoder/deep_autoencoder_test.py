from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

from keras.datasets import mnist
import numpy as np

""" Load data """
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print x_train.shape
print x_test.shape

""" Build model """
# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# input
input_img = Input(shape=(784,))

# Two parts: one for encoding and the other for decoding
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# We can form simple autoencoder - this one is for training
autoencoder = Model(input_img, decoded)

# We can also form coder for deploying
encoder = Model(input_img, encoded)

# And decoder
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = Dense(64, activation='relu')(encoded_input)
decoder_layer = Dense(128, activation='relu')(decoder_layer)
decoder_layer = Dense(784, activation='sigmoid')(decoder_layer)
decoder = Model(encoded_input, decoder_layer)

# Compile for training
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

""" Train """
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

""" Test """
# Encode and decode
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# Visualize results
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()