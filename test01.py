import tensorflow as tf
import numpy as np

# Verileri yükleme
data = np.load("veriler.npy")

# Verileri düzenleme
data = np.array(data)
data = data.reshape(-1, 28, 28, 1)

# Giriş ve çıkış katmanlarını tanımlama
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

# Modeli derleme
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Modeli eğitme
model.fit(data, epochs=10)


#This code block is a sample code that can be used to train an AI model. By giving cyber attack data to this model, a system that can detect cyber attacks can be developed. However, this is just a sample code block and further work and research may be required in developing a real cybersecurity system.
