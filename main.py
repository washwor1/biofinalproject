import playgame as pg
import numpy as np
import model as m
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])  # Set memory_limit in MB
    except RuntimeError as e:
        print(e)

model = m.getModel()
print(pg.play_game(model))


