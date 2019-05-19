#
# Emulator based on Gaussian Processes.
#

from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

from ._emulator import Emulator
import warnings
import numpy as np
import copy
import GPy


class GPEmulator(Emulator):
    """
    *Extends:* :class:`Emulator`

    Emulator using Gaussian Processes. This class provides interface with GPy.

    Arguments:

    ``log_likelihood``
        A :class:`LogPDF`, the likelihood distribution being emulated.
    ``X``
        N by n_paremeters matrix containing inputs for training data
    ``y``
        N by 1, target values for each input vector
    """

    def __init__(self, log_likelihood, X, y, **kwargs):
        super(GPEmulator, self).__init__(log_likelihood, X, y, **kwargs)

        # default model is Regression
        self.set_parameters(model=GPy.models.GPRegression)

    def __call__(self, x):
        assert hasattr(self, "_gp"), "Must first fit GP to data"

        # convert to np array
        if type(x) != np.ndarray:
            x = np.asarray(x)

        x = x.reshape((1, self._n_parameters))
        if self._input_scaler:
            x = self._input_scaler.transform(x)

        y = self._gp.predict_noiseless(x)[0]

        """
        TODO: include warnings?
        if y >= 0:
            warnings.warn("Non-negative log_likelihood predicted." +
                          "Indicative of high uncertainty in predictions.")
        """

        if self._output_scaler:
            y = self._output_scaler.inverse_transform(y)

        return y

    def predict(self, x, **kwargs):
        """
        Returns mean, var for given input parameters.
        """
        assert hasattr(self, "_gp"), "Must first fit GP to data"

        if self._input_scaler:
            x = self._input_scaler.transform(x)

        # don't apply output scaler to preserve variance values properly
        return self._gp.predict_noiseless(x, **kwargs)

    def set_parameters(
            self,
            model=None,
            kernel=None,
            optimizer=None):

        if model:
            self._model = model

        if kernel:
            self._kernel = kernel

        if optimizer:
            self._optimizer = optimizer

    def fit(self, optimize=True, messages=True, **kwargs):
        """
        Creates a GP for the provided data.
        **kwargs can include any additional arguments to GPy instance creation,
        e.g. normalizer = True normalizes the outputs.
        By default optimizes the GP, however this can be turned off to provide
        additional arguments for GPy optimizer.
        """
        if hasattr(self, '_kernel'):
            self._gp = self._model(self._X, self._y, self._kernel, **kwargs)
        else:
            self._gp = self._model(self._X, self._y, **kwargs)

        if optimize:
            self.optimize(messages=messages)

    def optimize(self, messages=True, **kwargs):
        """
        Optimize GP to data. **kwargs are the parameters for the GPy optimizer.
        """
        if hasattr(self, '_optimizer'):
            self._gp.optimize(self._optimizer, messages=messages, **kwargs)
        else:
            self._gp.optimize(messages=messages, **kwargs)

    def summary(self):
        print("Summary")
        print("Kernel:\n",
              self._kernel if hasattr(self, '_kernel') else "default")
        print("Model: ",
              self._model if hasattr(self, '_model') else "default")
        print("Optimizer: ",
              str(self._optimizer) if hasattr(self, '_optimizer')
              else "default")
        print(self._gp if hasattr(self, "_gp") else "No fit performed")

    def get_gp(self):
        """
        Returns a copy of GPy model
        """
        assert hasattr(self, "_gp"), "Must first fit GP"

        return copy.deepcopy(self._gp)

    def get_trained_kern(self):
        """
        Returns kernel currently used by gp
        """
        assert hasattr(self, "_gp"), "Must first fit GP"

        return self._gp.kern

    def get_log_marginal_likelihood(self):
        """
        Returns the log marginal likelihood of the model.
        """
        assert hasattr(self, "_gp"), "Must first fit GP"

        return self._gp.log_likelihood()
