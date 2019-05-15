#
# Base class for all emulators
#

from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np
import pints
import copy


class Emulator(pints.LogPDF):
    """
    *Extends:* :class:`LogPDF`

    Abstract class from which all emulators should inherit.
    An instance of the Emulator models given log-likelihood

    Arguments:

    ``log_likelihood``
        A :class:`LogPDF`, the likelihood distribution being emulated.
    ``X``
        N by n_paremeters matrix containing inputs for training data
    ``y``
        N by 1, target values for each input vector
    ``input_scaler``
        sklearn scalar type, don't pass class just the type.
        E.g. StandardScaler provides standardization.
    ``output_scaler``
        sklearn scaler class that will be applied to output
    """

    def __init__(self, log_likelihood, X, y,
                 input_scaler=False, output_scaler=False):
        # Perform sanity checks for given data
        if not isinstance(log_likelihood, pints.LogPDF):
            raise ValueError("Given pdf must extend LogPDF")

        self._n_parameters = log_likelihood.n_parameters()

        # check if dimensions are valid
        if X.ndim != 2:
            raise ValueError("Input should be 2 dimensional")

        X_r, X_c = X.shape

        if (X_c != self._n_parameters):
            raise ValueError("Input data should have n_parameters features")

        # if given target array is 1d convert automatically
        if y.ndim == 1:
            y = y.reshape(len(y), 1)

        if y.ndim != 2:
            raise ValueError("Target array should be 2 dimensional (N, 1)")

        y_r, y_c = y.shape

        if y_c != 1:
            raise ValueError("Target array should only have 1 feature")

        if (X_r != y_r):
            raise ValueError("Input and target dimensions don't match")

        # Normalize data for inputs and output
        # need to fit to test data
        self._input_scaler = input_scaler
        if input_scaler:
            self._input_scaler.fit(X)

        self._output_scaler = output_scaler
        if output_scaler:
            self._output_scaler.fit(y)

        # copy input data to avoid possible changes to it outside the class
        if input_scaler:
            self._X = self._input_scaler.transform(X)
        else:
            self._X = copy.deepcopy(X)

        if output_scaler:
            self._y = self._output_scaler.transform(y)
        else:
            self._y = copy.deepcopy(y)

    def n_parameters(self):
        return self._n_parameters
