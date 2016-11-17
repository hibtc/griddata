"""
Generate particles from a given probability distribution where each particle
is assigned a value of one. Then interpolate between the found data points
with `griddata`. For comparison, search regions far away from any found points
and insert zero values at these locations. Interpolate with this modification.
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

from util import plot2d, trace
from interpol import (
    Grid, Box, far_points__weighted_cumulative,
    scatter, generate_particle,
)


def mysterious_prob_dist(x, y):
    x, y = x/35, y/35
    return 1 - np.cos(x - y**2)


def main():

    # grid for plotting
    pgrid = Grid(Box([0, 0], [1, 1]), 100)
    plot_radius = 0.05
    nozero_radius = 0.025

    # grid for interpolation
    igrid = Grid(pgrid.box, 50)

    # probability distribution

    with trace("Generate probability distribution"):
        orig_pdist = np.fromfunction(mysterious_prob_dist, pgrid.shape)

    with trace("Plotting probability distribution"):
        plot2d(orig_pdist)
    plt.show()

    # particle scatter

    with trace("Generate particles"):
        points = np.array([
            generate_particle(orig_pdist)
            for i in range(500)
        ]) / (pgrid.shape - 1)
        values = np.ones(len(points))
        widths = np.ones(len(points))

    with trace("Generating particle scatter"):
        plotdata = scatter(pgrid, points, plot_radius)

    with trace("Plotting particle scatter"):
        plot2d(plotdata)
    plt.show()

    # zeros scatter

    with trace("Computing zeros"):
        zero_points = far_points__weighted_cumulative(
            igrid, points, values, widths, nozero_radius)
        zero_values = np.zeros(len(zero_points))

    with trace("Generating zeros scatter"):
        plotdata = scatter(pgrid, zero_points, plot_radius)

    with trace("Plotting zeros scatter"):
        plot2d(plotdata)
    plt.show()

    # without zeros

    with trace("Interpolating without zeros"):
        interpolate_naive = scipy.interpolate.griddata(
            points, values,
            pgrid.xi(), fill_value=0)

    with trace("Plotting interpolation without zeros"):
        plot2d(interpolate_naive.reshape(pgrid.shape))
    plt.show()

    # with zeros

    with trace("Interpolating with zeros"):
        interpolate_zeros = scipy.interpolate.griddata(
            np.vstack((points, zero_points)),
            np.hstack((values, zero_values)),
            pgrid.xi(), fill_value=0)

    with trace("Plotting interpolation with zeros"):
        plot2d(interpolate_zeros.reshape(pgrid.shape))
    plt.show()


if __name__ == '__main__':
    main()
