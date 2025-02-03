import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
# X_train = X_train / 255
# X_test = X_test / 255
# X_train_flattened = X_train.reshape(len(X_train), 28*28)
# X_test_flattened = X_test.reshape(len(X_test), 28*28)

# Reshape and normalize the data
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Define the CNN model
model = keras.Sequential(
    [
        keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train the model
# model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
# model = keras.Sequential(
#     [
#         keras.layers.Dense(100, activation="relu"),
#         keras.layers.Dense(10, activation="sigmoid"),
#     ]
# )

# model.compile(
#     optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
# )

# model.fit(X_train_flattened, y_train, epochs=5)
model.fit(X_train, y_train, epochs=5)
# Save the model to a file
model.save("mnist_cnn.keras")
print("Model saved to mnist_cnn.keras")
