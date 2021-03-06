#
# Plotting functions for emulator related problems
#

from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

from . import utils as emutils
import numpy as np


def surface(
        x_grid, y_grid, z_grid,
        cmap="Blues", angle=(25, 300), alpha=1.,
        fontsize=14, labelpad=10,
        title="", x_label="", y_label="", z_label="log_likelihood"):
    """
    Creates 3d contour plot given a grid for each axis.

    Arguments:

    ``x_grid``
        An NxN grid of values.
    ``y_grid``
        An NxN grid of values.
    ``z_grid``
        An NxN grid of values. z_grid determines colour.
    ``cmap``
        (Optional) Colour map used in the plot
    ``angle``
        (Optional) tuple specifying the viewing angle of the graph
    ``alpha``
        (Optional) alpha parameter of the surface
    ``fill``
        (Optional) Used to specify whether or not contour plot should be filled
        Default False.
    ``fontsize``
        (Optional) the fontsize used for labels
    ``labelpad``
        (Optional) distance of axis labels from the labels
    ``x_label``
        (Optional) The label of the x-axis
    ``y_label``
        (Optional) The label of the y-axis
    ``z_label``
        (Optional) The label of the z-axis

    Returns a ``matplotlib`` figure object and axes handle.
    """
    import matplotlib.pyplot as plt
    ax = plt.axes(projection='3d')
    # Data for a three-dimensional line
    ax.plot_surface(x_grid, y_grid, z_grid, cmap=cmap, alpha=alpha)
    ax.view_init(*angle)

    fontsize = fontsize
    labelpad = labelpad

    if title:
        plt.title(title, fontsize=fontsize)
    if x_label:
        ax.set_xlabel(x_label, fontsize=fontsize, labelpad=labelpad)
    if y_label:
        ax.set_ylabel(y_label, fontsize=fontsize, labelpad=labelpad)
    if z_label:
        ax.set_zlabel(z_label, fontsize=fontsize, labelpad=labelpad)

    return ax


def contour(
        x_grid, y_grid, z_grid,
        cmap="Blues", fill=False,
        fontsize=14, labelpad=10,
        title="",
        x_label="", y_label=""):
    """
    Creates 3d contour plot given a grid for each axis.

    Arguments:

    ``x_grid``
        An NxN grid of values.
    ``y_grid``
        An NxN grid of values.
    ``z_grid``
        An NxN grid of values. z_grid determines colour.
    ``cmap``
        (Optional) Colour map used in the plot
    ``fill``
        (Optional) Used to specify whether or not contour plot should be filled
        Default False.
    ``fontsize``
        (Optional) the fontsize used for labels
    ``labelpad``
        (Optional) distance of axis labels from the labels
    ``x_label``
        (Optional) The label of x-axis
    ``y_label``
        (Optional) The label of y-axis

    Returns a ``matplotlib`` figure object and axes handle.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))

    if fill:
        axes.contourf(x_grid, y_grid, z_grid, cmap=cmap)
    else:
        axes.contour(x_grid, y_grid, z_grid, cmap=cmap)

    fontsize = fontsize
    labelpad = labelpad

    if title:
        plt.title(title, fontsize=fontsize)
    if x_label:
        axes.set_xlabel(x_label, fontsize=fontsize, labelpad=labelpad)
    if y_label:
        axes.set_ylabel(y_label, fontsize=fontsize, labelpad=labelpad)

    plt.tight_layout()
    return fig, axes


def confidence_interval(param_range, mean, conf, show_points=True):
    """
    Creates a plot

    Arguments:

    ``param_range``
        An arary of values
    ``x``
        A point in the function's input space.
    ``lower``
        (Optional) Lower bounds for each parameter, used to specify the lower
        bounds of the plot.
    ``upper``
        (Optional) Upper bounds for each parameter, used to specify the upper
        bounds of the plot.
    ``evaluations``
        (Optional) The number of evaluations to use in each plot.

    Returns a ``matplotlib`` figure object and axes handle.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))

    lower = mean - conf
    upper = mean + conf

    axes.plot(param_range, mean, color="black")
    axes.plot(param_range, upper, color="grey")
    axes.plot(param_range, lower, color="grey")
    axes.fill_between(param_range, lower, upper, color="lightgrey")

    if show_points:
        axes.scatter(param_range, mean)

    plt.tight_layout()
    return fig, axes


def plot_surface_fixed_param(log_likelihood,
                             bounds,
                             fixed=None,
                             n_splits=50,
                             index_to_param_name=None,
                             contour=True,
                             additional_log_likelihoods=None,
                             precision=5,
                             **kwargs
                             ):
    """
    2d contour or a 3d plot for high-dimensional model.
    Have to provide fixed values for some parameters.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = plt.axes(projection = None if contour else '3d')
    n_parameters = bounds.n_parameters()

    # if variables not named use lowercase alphabet
    if index_to_param_name is None:
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        var_names = alphabet[:log_likelihood.n_parameters()]
        index_to_param_name = dict(enumerate(var_names))

    # fix some parameters at random if not given
    if not fixed and n_parameters > 2:
        fixed = list(enumerate(bounds.sample(1)[0]))
        fixed.pop(np.random.randint(n_parameters))
        fixed.pop(np.random.randint(n_parameters - 1))

    # get indices of params that are not fixed
    if fixed:
        p1_idx, p2_idx = [i for i in range(bounds.n_parameters())
                          if i not in [j for (j, _) in fixed]]
    else:
        p1_idx, p2_idx = 0, 1

    # generate surfaces
    p1_grid, p2_grid, grid = emutils.generate_grid(
        bounds.lower(),
        bounds.upper(),
        n_splits,
        fixed=fixed
    )

    likelihood_prediction = emutils.predict_grid(log_likelihood, grid)

    # plotting
    if contour is True:
        ax.contourf(p1_grid, p2_grid,
                    likelihood_prediction,
                    cmap="Reds")
    else:
        ax.plot_surface(p1_grid, p2_grid,
                        likelihood_prediction,
                        alpha=0.9,
                        **kwargs
                        )
        # predict other provided surfaces
        if additional_log_likelihoods:
            for likelihood in additional_log_likelihoods:
                likelihood_prediction = emutils.predict_grid(
                    likelihood,
                    grid
                )
                ax.plot_surface(
                    p1_grid, p2_grid,
                    likelihood_prediction,
                    alpha=0.3,
                    **kwargs
                )

    # titles and fixed values
    if fixed:
        fixed_parameters_string = ", ".join(
            [("{}={:."+ str(precision) + "f}").format(
                index_to_param_name[i], val) for (i, val) in fixed
            ]
        )
        ax.set_title("Fixed:" + fixed_parameters_string)
    ax.set_xlabel(index_to_param_name[p1_idx])
    ax.set_ylabel(index_to_param_name[p2_idx])
    if not contour:
        ax.set_zlabel("Likelihood")

    plt.tight_layout()

    return fig, ax


def plot_fixed_param_grid(log_likelihood,
                          fixed_parameters,
                          bounds,
                          n_splits=50,
                          shape=None,
                          index_to_param_name=None,
                          contour=True,
                          additional_log_likelihoods=None,
                          **kwargs
                          ):
    """
    Creates a 2d contour or a 3d plot grid.
    For high dimensional model fix by taking the mid-point.

    Arguments:

    ``log_likelihood``
        A :class:`LogPDF`, the likelihood function to plot.
    ``fixed_parameters``
        An array of lists, where each element in a list is a tuple (i, val)
        where i shall be fixed to value val.
    ``bounds``
        A :class:`Bounds`, bounds for each parameter in log_likelihood
    ``n_splits``
        (Optional) Number of splits along each axis.
    ``shape``
        (Optional) shape = (rows, cols), the number of rows and
        columns in the grid.
        Should have: rows * cols = len(fixed_parameters)
    ``index_to_param_name``
        (Optional) Dictoinary mapping the index of parameter to its name
    ``contour``
        (Optional) If True draw 2d contour plot, otherwise 3d plot
    ``additional_log_likelihoods``
        (Optional) List of additional log_likelihoods to display.

    Returns a ``matplotlib`` figure object and axes handle.
    """
    import matplotlib.pyplot as plt

    # if variables not named use lowercase alphabet
    if index_to_param_name is None:
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        var_names = alphabet[:log_likelihood.n_parameters()]
        index_to_param_name = dict(enumerate(var_names))

    if shape:
        rows, cols = shape
    else:
        rows = len(fixed_parameters)
        cols = 1

    fig, axes = plt.subplots(rows, cols, figsize=(5 * rows, 5 * cols))

    for row in range(rows):
        for col in range(cols):
            # change of axes required for 3d plots
            if contour is False:
                axes[row, col].remove()
                axes[row, col] = fig.add_subplot(
                    rows,
                    cols,
                    row * cols + col + 1,
                    projection="3d"
                )

            ax = axes[row, col]

            # list of fixed values
            fixed = fixed_parameters[row * cols + col]

            # get indices of params that are not fixed
            p1_idx, p2_idx = [i for i in range(bounds.n_parameters())
                              if i not in [j for (j, _) in fixed]]

            # generate surfaces
            p1_grid, p2_grid, grid = emutils.generate_grid(
                bounds.lower(),
                bounds.upper(),
                n_splits,
                fixed=fixed
            )

            # switch to get row i parameter first
            if row > col:
                p1_idx, p2_idx = p2_idx, p1_idx
                p1_grid, p2_grid = p2_grid, p1_grid

            likelihood_prediction = emutils.predict_grid(log_likelihood, grid)

            # plotting
            if contour is True:
                ax.contourf(p1_grid, p2_grid,
                            likelihood_prediction,
                            cmap="Reds")
            else:
                ax.plot_surface(p1_grid, p2_grid,
                                likelihood_prediction,
                                alpha=0.8,
                                **kwargs
                                )
                if additional_log_likelihoods:
                    for likelihood in additional_log_likelihoods:
                        likelihood_prediction = emutils.predict_grid(
                            likelihood,
                            grid
                        )
                        ax.plot_surface(
                            p1_grid, p2_grid,
                            likelihood_prediction,
                            alpha=0.5,
                            **kwargs
                        )

            # set title of each subgraph as the values of fixed parameters
            fixed_parameters_string = ", ".join(
                ["{}={:.5f}".format(index_to_param_name[i], val) for (i, val) in fixed]
            )
            ax.set_title("Fixed:" + fixed_parameters_string)
            ax.set_xlabel(index_to_param_name[p1_idx])
            ax.set_ylabel(index_to_param_name[p2_idx])
            if not contour:
                ax.set_zlabel("Likelihood")

    plt.tight_layout()
    return fig, axes


def plot_history(history):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(
        history.epoch,
        np.array(history.history['mean_absolute_error']),
        label='Train Loss',
    )
    plt.plot(
        history.epoch,
        np.array(history.history['val_mean_absolute_error']),
        label='Val loss',
    )
    plt.tight_layout()
    plt.legend()

    return fig, axes
