"""
A simple example based on a static list of a few points.
"""

from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

from util import plot2d
from interpol import (
    Grid, Box, scatter,
    far_points__weighted_cumulative,
    far_points__weighted_individual,
)


def main():
    points = np.array([
        [0., 0.],
        [2., 2.],
        [0.5, 2.],
        [-1, 3],
    ])
    values = 1.
    widths = 1.
    radius = 0.5

    box = Box.from_points(points)
    grid = Grid.from_box_raster(box, 0.5)

    indiv = far_points__weighted_individual(grid, points, values, widths, radius)
    cumul = far_points__weighted_cumulative(grid, points, values, widths, radius)

    # show zeros
    plot = Grid(grid.box, grid.shape*10)
    dist = scatter(plot, points, values, widths, 0.2)
    plot2d(plot, dist)
    dist = (scatter(plot, points, values, widths, 0.2) -
            scatter(plot, indiv,  values, widths, 0.1))
    plot2d(plot, dist)


if __name__ == '__main__':
    main()
