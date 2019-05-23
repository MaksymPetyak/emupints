#
# Predefined simple models for neural network emulators
#

import tensorflow as tf
from tensorflow import keras

# General rules:
# use identity for the last layer (regression tasks)
# relu activation function
# Minimise squared error


def create_model(n_parameters, model_size):
    if model_size == 'small':
        return create_small_model(n_parameters)
    elif model_size == 'average':
        return create_average_model(n_parameters)
    elif model_size == 'large':
        return create_large_model(n_parameters)
    else:
        raise "No such model type"


def create_small_model(n_parameters):
    model = keras.Sequential()

    model.add(keras.layers.Dense(
        64,
        activation=tf.nn.leaky_relu,
        input_shape=(n_parameters,),
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
    ))
    model.add(keras.layers.Dense(
        128,
        activation=tf.nn.leaky_relu,
        kernel_regularizer=tf.keras.regularizers.l2(0.01),

    ))
    model.add(keras.layers.Dense(
        64,
        activation=tf.nn.leaky_relu,
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    ))
    model.add(keras.layers.Dense(1, activation=tf.identity))

    return model


def create_average_model(n_parameters):
    model = keras.Sequential()

    model.add(keras.layers.Dense(
        128,
        activation=tf.nn.leaky_relu,
        input_shape=(n_parameters,),
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    ))
    model.add(keras.layers.Dense(
        256,
        activation=tf.nn.leaky_relu,
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    ))
    model.add(keras.layers.Dense(
        128,
        activation=tf.nn.leaky_relu),
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )
    model.add(keras.layers.Dense(1, activation=tf.identity))

    return model


def create_large_model(n_parameters):
    model = keras.Sequential()

    model.add(keras.layers.Dense(
        64,
        activation=tf.nn.leaky_relu,
        input_shape=(n_parameters,),
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    ))
    model.add(keras.layers.Dense(
        128,
        activation=tf.nn.leaky_relu,
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    ))
    model.add(keras.layers.Dense(
        256,
        activation=tf.nn.leaky_relu,
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    ))
    model.add(keras.layers.Dense(
        128,
        activation=tf.nn.leaky_relu,
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    ))
    model.add(keras.layers.Dense(
        64,
        activation=tf.nn.leaky_relu,
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    ))
    model.add(keras.layers.Dense(1, activation=tf.identity))

    return model
