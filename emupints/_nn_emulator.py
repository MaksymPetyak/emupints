#
# Emulator based on Neural Networks.
#

from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

from ._emulator import Emulator
from .models import create_model
import warnings
import numpy as np
import copy

import tensorflow as tf
from tensorflow import keras


class NNEmulator(Emulator):
    """
    *Extends:* :class:`Emulator`

    Emulator using neural networks. This class provides interface with Keras.
    Can use the default models provided with arguments:
    'small', 'average', 'large'
    Alternatively can manually set model through set_model(method) method


    Arguments:

    ``log_likelihood``
        A :class:`LogPDF`, the likelihood distribution being emulated.
    ``X``
        N by n_parameters matrix containing inputs for training data
    ``y``
        N by 1, target values for each input vector
    ``normalize_input``
        If true then inputs will be normalized
    """

    def __init__(self, log_likelihood, X, y, model_size='average', **kwargs):
        super(NNEmulator, self).__init__(log_likelihood, X, y, **kwargs)

        # default model is Regression
        self._model = create_model(log_likelihood._n_parameters, model_size)

    def __call__(self, x):
        """
        Additional **kwargs can be provided to Keras's predict method
        """
        x = x.reshape((1, self.n_parameters()))

        if self._input_scaler:
            x = self._input_scaler.transform(x)

        # convert to np array
        if type(x) != np.ndarray:
            x = np.asarray(x)

        y = self._model.predict([x])

        if self._output_scaler:
            y = self._output_scaler.inverse_transform(y)

        return y

    def set_parameters(self, loss='mse', optimizer='adam',
                       metrics=['mae'], **kwargs):
        """
        Provide parameters to compile the model
        """
        self._model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,  # usually mae
            **kwargs
        )

    def fit(self, epochs=50, batch_size=32, validation_split=0.2, **kwargs):
        """
        Training neural network and return history
        """
        history = self._model.fit(
                    self._X,
                    self._y,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    **kwargs
                   )

        # save to return in the future
        self._history = history

        return history

    def summary(self):
        return self._model.summary()

    def evaluate(self, x, **kwargs):
        """
        Uses Keras's evaluate() method, so can provide additional paramaters
        """
        return self._model.evaluate(x, **kwargs)

    def get_model(self):
        """
        Returns model
        """
        return self._model

    def get_model_history(self):
        """
        Returns the log marginal likelihood of the model.
        """
        assert hasattr(self, "_history"), "Must first train NN"

        return self._history
