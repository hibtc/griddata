"""
Analysis utils for some pepperpot data.

Usage:
    pepperpot [OPTIONS] filename
    pepperpot (-h | -v)

Options:
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

import tempfile
import sys

import numpy as np
import scipy.spatial
import scipy.interpolate
import matplotlib.pyplot as plt

from docopt import docopt

from util import plot2d, trace
from interpol import (
    Grid, Box, far_points__weighted_cumulative,
    scatter, generate_particle,
    restrict_to_polytope,
)


COL_TITLES = ('x', 'y', 'px', 'py')
PLOTS_2D = ([0, 1], [0, 2], [1, 3],
            [2, 3], [0, 3], [1, 2])


def get_columns(array, columns):
    if isinstance(columns, list):
        block = array[columns]
        return block.view(np.float64).reshape(block.shape + (-1,))
    else:
        return array[columns]


def read_ppp(in_file):
    rawdata = np.genfromtxt(in_file, names=True)
    points = get_columns(rawdata, ['x', 'y', 'xprime', 'yprime'])
    values = get_columns(rawdata, 'weight')
    return points, values


def interpolate_pdist(points, values, widths, igrid, zgrid, radius):
    with trace("Computing zeros"):
        zero_points = far_points__weighted_cumulative(
            zgrid, points, values, widths, radius)
    print("  Number of zero points: {}".format(len(zero_points)))

    with trace("Restricting to convex hull"):
        hull = scipy.spatial.ConvexHull(np.array(points))
        zero_points = restrict_to_polytope(hull.equations, zero_points)
    print("  Relative hull volume: {}".format(hull.volume / zgrid.box.volume))
    print("  Selected zero points: {}".format(len(zero_points)))

    with trace("Interpolating 4D probability distribution"):
        zero_values = np.zeros(len(zero_points))
        pdist = scipy.interpolate.griddata(
            np.vstack((points, zero_points)),
            np.hstack((values, zero_values)),
            igrid.xi(), fill_value=0)
        return pdist.reshape(igrid.shape)


def save_pdist(pdist, out_file=None):
    """Save a probability distribution matrix to a file."""
    if out_file is None:
        out_file = tempfile.mktemp(prefix='pdist_', dir='.')
    with trace("Saving probability distribution to: {}".format(out_file)):
        np.save(out_file, pdist)


def plot_gauss_sum(grid_4d, points, values, widths, radius_4d):
    """Plot the 2D projections of a 4D particle scatter."""
    plt.clf()
    for i, comb in enumerate(PLOTS_2D):
        title = '{}/{}'.format(*(COL_TITLES[c] for c in comb))
        ax = plt.subplot(2, 3, i+1)
        ax.set_title(title)
        # projected plot grid
        ppoints = points[:,comb]
        pgrid = Grid(grid_4d.box.projection(comb), 100)
        with trace("Generate {} particle scatter".format(title)):
            pdata = scatter(pgrid, ppoints, values, widths, radius_4d[comb])
        with trace("Plotting {} particle scatter".format(title)):
            plot2d(pdata)
    plt.show()


def plot_pdist(pdist):
    """Plot the 2D projections of a 4D probability distribution matrix."""
    plt.clf()
    for i, comb in enumerate(PLOTS_2D):
        title = '{}/{}'.format(*(COL_TITLES[c] for c in comb))
        ax = plt.subplot(2, 3, i+1)
        ax.set_title(title)
        with trace("Projecting pdist along {}".format(title)):
            along = set(range(4)) - set(comb)
            pdata = np.sum(pdist, axis=tuple(along))
        with trace("Plotting pdist along   {}".format(title)):
            plot2d(pdata)
    plt.show()


def main():

    points, values = read_ppp(in_file)
    widths = 1

    # grid for interpolation:
    # 25
    zgrid = Grid(Box.from_points(points), 15)
    igrid = Grid(zgrid.box, 15)

    # box size for averaging = 2 mm | 4 mrad
    #radius = np.array([2., 2., 4., 4.])
    radius = np.array([4., 4., 6., 6.])

    plot_scatter(points, igrid, radius)

    pdist = interpolate_pdist(points, values, widths, igrid, zgrid, radius)
    save_pdist(pdist, out_file)
    plot_pdist(pdist)


if __name__ == '__main__':
    main()
