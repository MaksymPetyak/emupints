#
# Predefined problems that can be quickly loaded for experiments
#

import pints
import pints.toy as toy

from . import utils as emutils
import numpy as np

import copy


class Problems():
    """
    Class containing parameters for the problems used.
    load_problem() method returns a dictionary with instantiated model
    """
    LogisticModel = {
        'model': toy.LogisticModel,     # only class, not an instance
        'n_parameters': 2,
        'parameters': np.array([0.15, 500]),
        'n_outputs': 1,
        'times': np.linspace(0, 100, 100),
        'simulation_noise_percent': 0.05,   # std in normally distributed noise
        'param_names': ['r', 'K'],
        'param_range': [[0.1, 400], [0.2, 600]],    # lower and upper bounds
        'prior': pints.UniformLogPrior,
    } # continious

    LotkaVolterraModel = {
        'model': toy.LotkaVolterraModel,
        'n_parameters': 4,
        'parameters': np.array([3, 2, 3, 2]),
        'n_outputs': 2,
        'times': np.linspace(0, 3, 300),
        'param_names': ['a', 'b', 'c', 'd'],
        'simulation_noise_percent': 0.05,
        'param_range_percent': 0.4,
        'prior': pints.UniformLogPrior,
    } # continious

    LotkaVolterraModelDiscontinious = {
        'model': toy.LotkaVolterraModel,
        'n_parameters': 4,
        'parameters': np.array([3, 2, 3, 2]),
        'n_outputs': 2,
        'times': np.linspace(0, 3, 300),
        'param_names': ['a', 'b', 'c', 'd'],
        'simulation_noise_percent': 0.05,
        'param_range': [[0, 0, 0, 0], [5, 5, 5, 5]],
        'prior': pints.UniformLogPrior,
    } # discontinious

    FitzhughNagumoModel = {
        'model': toy.FitzhughNagumoModel,
        'n_parameters': 3,
        'parameters': np.array([0.1, 0.5, 3]),
        'n_outputs': 2,
        'param_names': ['a', 'b', 'c', 'd'],
        'times': np.linspace(0, 20, 200),
        'simulation_noise_percent': 0.05,
        'param_range_percent': 0.5,
        'prior': pints.UniformLogPrior,
    } #continious

    FitzhughNagumoModelDiscontinious = {
        'model': toy.FitzhughNagumoModel,
        'n_parameters': 3,
        'parameters': np.array([0.1, 0.5, 3]),
        'n_outputs': 2,
        'param_names': ['a', 'b', 'c', 'd'],
        'times': np.linspace(0, 20, 200),
        'simulation_noise_percent': 0.05,
        'param_range': [[0, 0, 2], [1, 1, 4]],
        'prior': pints.UniformLogPrior,
    } # discontinious

    SIRModel = {
        'model': toy.SIRModel,
        'n_parameters': 3,
        'parameters': np.array([0.026, 0.285, 38]),
        'n_outputs': 2,
        'param_names': ['S', 'I', 'R'],
        'times': np.arange(1, 22),
        'simulation_noise_percent': 0.05,
        'param_range_percent': 0.5,
        'prior': pints.UniformLogPrior,
    }

    HodgkinHuxleyIKModel = {
        'model': toy.HodgkinHuxleyIKModel,
        'n_parameters': 5,
        'parameters': np.array([0.01, 10, 10, 0.125, 80]),
        'n_outputs': 1,
        'param_names': ['a', 'b', 'c', 'd', 'e'],
        'times': np.arange(0, 1200, 0.1),
        'simulation_noise_percent': 0.05,
        'param_range_percent': 0.2,
        'prior': pints.UniformLogPrior,
    }

    GoodwinOscillatorModel = {
        'model': toy.GoodwinOscillatorModel,
        'n_parameters': 5,
        'parameters': np.array([2, 4, 0.12, 0.08, 0.1]),
        'n_outputs': 3,
        'param_names': ['a', 'b', 'c', 'd', 'e'],
        'times': np.linspace(0, 100, 200),
        'simulation_noise_percent': 0.05,
        'param_range_percent': 0.2,
        'prior': pints.UniformLogPrior,
    }

    @staticmethod
    def load_problem(problem_dict):
        """
        Returns a dictionary containing an instantiated PINTS problem
        """
        problem_instance = copy.deepcopy(problem_dict)

        model = problem_dict["model"]()
        parameters = problem_dict['parameters']

        # simulate problem
        if 'simulation_noise_percent' in problem_dict:
            values, times, noise_stds = emutils.simulate(
                model,
                parameters=problem_dict['parameters'],
                times=problem_dict['times'],
                noise_range_percent=problem_dict['simulation_noise_percent'],
            )
        else:
            values, times = emutils.simulate(
                model,
                parameters=problem_dict['parameters'],
                times=problem_dict['times'],
                noise_range_percent=None,
            )
            noise_stds = None

        # create instance of a problem
        if problem_dict['n_outputs'] == 1:
            problem = pints.SingleOutputProblem(model, times, values)
        else:
            problem = pints.MultiOutputProblem(model, times, values)

        # create likelihood with or without known noise
        if noise_stds is not None:
            log_likelihood = pints.KnownNoiseLogLikelihood(problem, noise_stds)
        else:
            log_likelihood = pints.UnknownNoiseLogLikelihood(problem)

        # should either provide the percentage range for parameters
        # or the parameter range itself
        if 'param_range_percent' in problem_dict:
            param_range_percent = problem_dict['param_range_percent']
            params_lower = parameters - param_range_percent * np.abs(parameters)
            params_upper = parameters + param_range_percent * np.abs(parameters)
        else:
            params_lower, params_upper = problem_dict['param_range']

        bounds = pints.RectangularBoundaries(
            lower=params_lower,
            upper=params_upper
        )

        log_prior = problem_dict['prior'](bounds)
        log_posterior = pints.LogPosterior(log_likelihood, log_prior)

        # extend the dictionary with created variables
        problem_instance.update({
            'model': model,
            'values': values,
            'times': times,
            'noise_stds': noise_stds,
            'problem': problem,
            'bounds': bounds,
            'log_likelihood': log_likelihood,
            'log_prior': log_prior,
            'log_posterior': log_posterior
        })

        return problem_instance
