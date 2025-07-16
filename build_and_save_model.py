import tensorflow as tf
from tensorflow.keras import layers, models

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Build the model using updated Keras API
model = models.Sequential([
    layers.Input(shape=(28, 28)),          # Proper Input layer (no batch_input_shape)
    layers.Reshape((28, 28, 1)),           # Reshape input to match Conv2D expectation
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model briefly (or more if needed)
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Save the model in latest `.keras` format (safe to load later)
model.save("digit_recognizer_model.keras")