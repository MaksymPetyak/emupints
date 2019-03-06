import tensorflow as tf
from tensorflow import keras

# General rules:
# use identity for the last layer (regression tasks)
# relu activation function
# Minimise squared error


def create_small_model(n_parameters):
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(n_parameters,)))
    model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.identity))
    # shape [1, 10, 50, 10, 1]
    return model


def create_average_model(n_parameters):
    model = keras.Sequential()

    model.add(keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(n_parameters,)))
    model.add(keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(keras.layers.Dense(32, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.identity))

    return model


def create_large_model(n_parameters):
    model = keras.Sequential()

    model.add(keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(n_parameters,)))
    model.add(keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.identity))

    return model