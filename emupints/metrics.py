#
# Metrics used for comparing emulator performance
#

import numpy as np


def mae(y_true, y_pred):
    """
    Mean absolute error.
    Use for comparing predictions of real likelihood with emulated likelihood.
    """
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    """
    Mean absolute percentage error
    Measure of how successfull inference process has been
    Good result usually have mape < 0.05
    """
    return np.mean(np.abs((y_true - y_pred) / y_true))


def chain_mae(chain, emu_log_posterior, log_posterior):
    """
    Calculate the error in predictions along one chain
    """
    emu_prediction = np.apply_along_axis(emu_log_posterior, 1, chain).flatten()
    real_prediction = np.apply_along_axis(log_posterior, 1, chain).flatten()
    return mae(emu_prediction, real_prediction)


def chain_mape(chain, emu_log_posterior, log_posterior):
    """
    Calculate the error in predictions along one chain
    """
    emu_prediction = np.apply_along_axis(emu_log_posterior, 1, chain).flatten()
    real_prediction = np.apply_along_axis(log_posterior, 1, chain).flatten()
    return mape(emu_prediction, real_prediction)


def estimate_parameters(chains):
    parameters = np.mean(np.mean(chains, axis=1), axis=0)
    std = np.std(np.std(chains, axis=1), axis=0)
    return parameters, std

