import playgame as pg
import numpy as np
import model as m
import tensorflow as tf

print(f"Tensorflow Version: {tf.__version__} \nGPUs Running: {len(tf.config.list_physical_devices('GPU'))}\n{tf.test.gpu_device_name()}")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])  # Set memory_limit in MB
    except RuntimeError as e:
        print(e)
input_shape = (880, 1000, 4)
model = m.create_cnn_lstm_model(input_shape)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.summary()
print(pg.play_game(model))


