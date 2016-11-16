
from __future__ import division

import contextlib
import itertools
import functools
import operator
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate



def where(x):
    """Return list of index vectors where array is nonzero."""
    return np.transpose(np.nonzero(x))

def array_ceil(x):
    """Round array up and return as integer array."""
    return np.asarray(np.ceil(x), dtype=int)

def array_round(x):
    """Round array and return as integer array."""
    return np.asarray(np.round(x), dtype=int)


# TODO: smooth ellipses by integrating value in the square

def elliptic_distance(grid, ellipse_center, ellipse_shape):
    # Normalize to unit box:
    grid_shape = grid.cells
    grid_extent = grid.box.size - grid.raster
    ellipse_center = (ellipse_center - grid.min_point) / grid_extent
    ellipse_shape = ellipse_shape / grid_extent
    # collect contributions for each dimension
    mesh = np.meshgrid(*(
        ((x-e_center)/e_size) ** 2
        for g_size, e_center, e_size
        in zip(grid_shape, ellipse_center, ellipse_shape)
        for x in [np.linspace(0, 1, g_size)]
    ))
    return np.sum(mesh, axis=0)


def sum_(values, initial=0):
    return functools.reduce(operator.add, values, initial)


def unit_shape(dim, axis, size):
    return (1,) * (axis) + (size,) + (1,) * (dim-axis-1)


def normal_distribution(grid, ellipse_center, ellipse_shape):
    # Normalize to unit box:
    grid_shape = grid.cells
    grid_extent = grid.box.size - grid.raster
    ellipse_center = (ellipse_center - grid.min_point) / grid_extent
    ellipse_shape = ellipse_shape / grid_extent
    # collect contributions for each dimension
    axis_contributions = (
        (x-e_center)**2/(2*e_size**2)
        for g_size, e_center, e_size
        in zip(grid_shape, ellipse_center, ellipse_shape)
        for x in [np.linspace(0, 1, g_size)]
    )
    # can't use the following because meshgrid returns a weird transpose:
    #   collected = np.sum(np.meshgrid(*contributions), axis=0)

    # Instead, making use of numpy's broadcasting:
    collected = sum_(
        contrib.reshape(unit_shape(grid.box.dim, axis, len(contrib)))
        for axis, contrib in enumerate(axis_contributions))
    return np.exp(-collected)


def row_scalar(val):
    """Coerce array to row vector, leave scalar as is."""
    return val if np.isscalar(val) else np.asarray(val)

def col_scalar(val):
    """Coerce array to col vector, leave scalar as is."""
    return val if np.isscalar(val) else np.asarray(val)[:,None]

def row_vector(val, dim):
    """Broadcast to row vector."""
    return np.ones(dim) * row_scalar(val)

def col_vector(val, dim):
    """Broadcast to row vector."""
    return np.ones((dim, 1)) * col_scalar(val)


class Box(object):

    """
    Hyper-rectangle / n-dimensional box that is aligned with the coordinate
    axes.

    :ivar min_bound:    [np.ndarray] minimum coordinate (bottom left)
    :ivar max_bound:    [np.ndarray] maximum coordinate (top right)
    :ivar size:         [np.ndarray] box size in coordinate units
    :ivar int dim:      dimension (size of a coordinate vector)
    """

    def __init__(self, min_bound, max_bound, dim=None):
        self.dim = len(min_bound) if dim is None else dim
        self.min_bound = row_vector(min_bound, self.dim)
        self.max_bound = row_vector(max_bound, self.dim)
        self.size = self.max_bound - self.min_bound

    @classmethod
    def from_points_fuzzy(cls, points, values, radius):
        """
        Find min/max coordinates with additional space at edges for ellipses
        around the individual points with half-axes weighted.

        :param points:  [np.ndarray]        coordinate vectors
        :param values:  [np.ndarray]        point's ellipse half-axes weights
        :param radius:  [np.ndarray|float]  ellipse's half-axes coefficients
        """
        points = np.asarray(points)
        values = col_scalar(values)
        radius = row_scalar(radius)
        return cls(np.min(points-values*radius, axis=0),
                   np.max(points+values*radius, axis=0))


class Grid(object):

    """
    Holds information about a discretization of an orthogonal box. The box is
    divided into a number cells whose coordinates are located at the center.

    :ivar Box box:  the coordinate bounds
    :ivar cells:    [np.ndarray] number of cells in each direction
    :ivar raster:   [np.ndarray] cell size in each direction
    """

    def __init__(self, box, cells):
        self.box = box
        self.cells = np.asarray(row_vector(cells, box.dim), dtype=int)
        self.raster = box.size / cells
        self.min_point = self.box.min_bound + self.raster/2
        self.max_point = self.box.max_bound - self.raster/2
        self.min_index = np.zeros(self.box.dim, dtype=int)
        self.max_index = self.cells

    @classmethod
    def from_box_raster(cls, box, raster):
        """
        Create grid from a box and a given approximate raster size. The
        :class:`Grid` instance will hold the real raster size.
        """
        return cls(box, array_ceil(box.size/raster))

    def index_to_point(self, index):
        """Convert index in the represented mesh to coordinate point."""
        return self.min_point + index * self.raster

    def point_to_index(self, point):
        """Convert coordinate point to index in the represented mesh."""
        point = np.asarray(point)
        if point.ndim == 2:
            return np.array([self.point_to_index(p) for p in point])
        return np.asarray(
            np.clip(np.round((point - self.min_point) / self.raster),
                    self.min_index,
                    self.max_index),
            dtype=int)


def zeros_for_interpolation_weighted_cumulative(
        grid, points, values, radius, threshold=1/np.exp(2)):

    points = np.asarray(points)
    values = row_vector(values, points.shape[0])

    # place an normal distribution around each point with the radius weighted
    # by the corresponding intensity
    dists = (normal_distribution(grid, point, value*radius)
             for point, value in zip(points, values))

    # sum up contributions
    cumulative = sum_(dists)
    zero_mask = cumulative <= threshold

    # TODO: select only those points in the convex hull

    zero_indices = where(zero_mask)
    zero_points = grid.index_to_point(zero_indices)
    return zero_points


def zeros_for_interpolation_weighted_individual(
        grid, points, values, radius, threshold=1):

    points = np.asarray(points)
    values = row_vector(values, points.shape[0])

    # place an ellipse around each point with the radius weighted by the
    # corresponding intensity
    dists = [elliptic_distance(grid, point, value*radius)
             for point, value in zip(points, values)]

    # generate masks for individual points and compute their disjunction
    individual = np.array(dists) <= threshold
    zero_mask = np.any(individual, axis=0)

    # TODO: select only those points in the convex hull

    zero_indices = where(zero_mask)
    zero_points = grid.index_to_point(zero_indices)
    zero_values = np.zeros((len(zero_points), 1))
    return (zero_indices, zero_values)


def generate_particle(pdist):
    result = np.empty(0, dtype=int)
    while isinstance(pdist, np.ndarray):
        # determine probabilities for finding a particular value of X, where X
        # is the first left-over dimension in the probability distribution:
        yz_axes = tuple(range(1, pdist.ndim))
        x_weights = np.sum(pdist, axis=yz_axes)
        x_pdist = x_weights / np.sum(x_weights)
        # generate value for X and append to result coordinate vector:
        coord = np.random.choice(len(x_pdist), p=x_pdist)
        result = np.hstack((result, coord))
        # restrict probability distribution to the given case:
        pdist = pdist[coord]
    return result


def scatter(grid, points, radius):
    # use `sum_(())` instead of `np.sum([])` to avoid huge memory bloat (!)
    return sum_(normal_distribution(grid, point, radius)
                for point in points)


def plot2d(image):
    plt.imshow(image)
    return plt


def special_function(x, y):
    x, y = x/35, y/35
    return 1 - np.cos(x - y**2)


@contextlib.contextmanager
def trace(message):
    print(message)
    start = time.time()
    try:
        yield None
    finally:
        stop = time.time()
        print("  .. {:.3f}s".format(stop - start))


def main():
    # grid for plotting
    min_bound = np.array([0, 0])
    max_bound = np.array([1, 1])
    num_cells = 100
    plotgrid = Grid(Box(min_bound, max_bound), num_cells)
    plot_radius = 0.05
    nozero_radius = 0.025

    interpolate_grid = Grid(Box(min_bound, max_bound), 50)

    xi = np.array(list(
        itertools.product(*(range(int(num)) for num in plotgrid.cells))
    )) / plotgrid.cells

    # probability distribution

    with trace("Generate probability distribution"):
        orig_pdist = np.fromfunction(special_function, plotgrid.cells)

    with trace("Plotting probability distribution"):
        plot2d(orig_pdist)
    plt.show()

    # particle scatter

    with trace("Generate particles"):
        points = np.array([
            generate_particle(orig_pdist)
            for i in range(500)
        ]) / plotgrid.cells
        values = np.ones(len(points))

    with trace("Generating particle scatter"):
        plotdata = scatter(plotgrid, points, plot_radius)

    with trace("Plotting particle scatter"):
        plot2d(plotdata)
    plt.show()

    # zeros scatter

    with trace("Computing zeros"):
        zero_points = zeros_for_interpolation_weighted_cumulative(
            interpolate_grid, points, values, nozero_radius)
        zero_values = np.zeros(len(zero_points))

    with trace("Generating zeros scatter"):
        plotdata = scatter(plotgrid, zero_points, plot_radius)

    with trace("Plotting zeros scatter"):
        plot2d(plotdata)
    plt.show()

    # without zeros

    with trace("Interpolating without zeros"):
        interpolate_naive = scipy.interpolate.griddata(
            points, values,
            xi, fill_value=0)

    with trace("Plotting interpolation without zeros"):
        plot2d(interpolate_naive.reshape(plotgrid.cells))
    plt.show()

    # with zeros

    with trace("Interpolating with zeros"):
        interpolate_zeros = scipy.interpolate.griddata(
            np.vstack((points, zero_points)),
            np.hstack((values, zero_values)),
            xi, fill_value=0)

    with trace("Plotting interpolation with zeros"):
        plot2d(interpolate_zeros.reshape(plotgrid.cells))
    plt.show()

    return


    points = np.array([
        [0., 0.],
        [2., 2.],
        [0.5, 2.],
    ])
    values = 1.
    radius = 0.5

    box = Box.from_points_fuzzy(points, values, radius)
    grid = Grid.from_box_raster(box, 0.2)

    indiv = zeros_for_interpolation_weighted_individual(grid, points, values, radius)
    cumul = zeros_for_interpolation_weighted_cumulative(grid, points, values, radius)

    # show zeros
    plot = Grid(grid.box, grid.cells*10)
    dist = scatter(plot, cumul[0], 0.1) - scatter(plot, points, 0.2)
    plot2d(dist)


if __name__ == '__main__':

    main()

