# A simple wrapper you can use for a created callable model

import pints


class EmulatorWrapper(pints.LogPDF):
    """
    *Extends:* :class:`pints.LogPDF`

    Wrapper class that allows to pints to treat a given function as a LogPDF
    """

    def __init__(self, emu, n_parameters):
        self._emu = emu
        self._n_parameters = n_parameters

    def __call__(self, x):
        return self._emu(x)

    def evaluateS1(self, x):
        pass

    def n_parameters(self):
        return self._n_parameters
