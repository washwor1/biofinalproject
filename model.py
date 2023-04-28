import tensorflow as tf

# Define the input layer.
input_layer = tf.keras.layers.Input(shape=(440, 500, 4))

# Define the convolutional layers.
conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(conv1)
conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(conv2)

# Define the pooling layers.
pool1 = tf.keras.layers.MaxPool2D((2, 2))(conv3)
pool2 = tf.keras.layers.MaxPool2D((2, 2))(pool1)

# Define the recurrent layers.
lstm1 = tf.keras.layers.LSTM(128)(pool2)
lstm2 = tf.keras.layers.LSTM(64)(lstm1)

# Define the output layer.
output_layer = tf.keras.layers.Dense(9, activation='softmax')(lstm2)

# Define the model.
model = tf.keras.models.Model(input_layer, output_layer)

# Compile the model.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()
