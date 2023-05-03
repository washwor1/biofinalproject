import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, Reshape
from tensorflow.keras.models import Model

def create_cnn_lstm_model(input_shape, output_size=15, max_vram=2):
    # Calculate the size of the LSTM layer based on the maximum VRAM constraint
    # lstm_units = int((max_vram * 1024**3) / (4 * input_shape[0] * input_shape[1]))

    # Input layer for the image
    input_img = Input(shape=input_shape)

    # CNN layers
    x = MaxPooling2D(pool_size=(2, 2))(input_img)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Flatten the CNN output
    x = Flatten()(x)

    # Reshape the flattened output to a sequence
    x = Reshape((1, -1))(x)

    # LSTM layer with the calculated number of units
    x = LSTM(32, activation='tanh')(x)

    # Dense output layer
    output = Dense(output_size, activation='linear')(x)

    # Create the model
    model = Model(input_img, output)

    return model

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# input_shape = (880, 1000, 4)
# model = create_cnn_lstm_model(input_shape)
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# model.summary()
