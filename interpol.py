"""
Utilities for working with numpy arrays and interpolation.
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

import itertools
import functools
import operator

import numpy as np
import scipy.ndimage


#----------------------------------------
# Utility functions
#----------------------------------------

def where(x):
    """Return list of index vectors where array is nonzero."""
    return np.transpose(np.nonzero(x))

def array_ceil(x):
    """Round array up and return as integer array."""
    return array_toint(np.ceil(x))

def array_round(x):
    """Round array and return as integer array."""
    return array_toint(np.round(x))

def sum_(values, initial=0):
    """Use `sum_(())` where `np.sum([])` would cause memory issues."""
    return functools.reduce(operator.add, values, initial)

def product_(values, initial=1):
    """Use `product_(())` where `np.product([])` would cause memory issues."""
    return functools.reduce(operator.mul, values, initial)

def unit_shape(dim, axis, size):
    """Return the shape of an 1D vector along the specified dimension."""
    return (1,) * (axis) + (size,) + (1,) * (dim-axis-1)

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

def array_toint(val):
    return np.asarray(val, dtype=int)


def _prep(points, values, widths):
    points = np.asarray(points)
    values = row_vector(values, points.shape[0])
    widths = row_vector(widths, points.shape[0])
    return points, values, widths


#----------------------------------------
# Gridding
#----------------------------------------

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
    def from_points(cls, points):
        """
        Find min/max coordinates with additional space at edges for ellipses
        around the individual points with half-axes weighted.
        """
        points = np.asarray(points)
        return cls(np.min(points, axis=0),
                   np.max(points, axis=0))

    def projection(self, axes):
        axes = list(axes)
        return self.__class__(self.min_bound[axes],
                              self.max_bound[axes])

    @property
    def volume(self):
        return np.product(self.size)


class Grid(object):

    """
    Holds information about a discretization of an orthogonal box. The box is
    divided into a number cells whose coordinates are located at the center.

    :ivar Box box:  the coordinate bounds
    :ivar shape:    [np.ndarray] number of sampling points in each direction
    :ivar raster:   [np.ndarray] space between adjacent sampling points each direction
    """

    def __init__(self, box, shape):
        self.box = box
        self.shape = array_toint(row_vector(shape, box.dim))
        self.raster = box.size / (shape - 1)
        self.min_index = np.zeros(box.dim, dtype=int)
        self.max_index = self.shape - 1

    @classmethod
    def from_box_raster(cls, box, raster):
        """
        Create grid from a box and a given approximate raster size. The
        :class:`Grid` instance will hold the real raster size.
        """
        return cls(box, array_ceil(box.size/raster))

    def subgrid(self, box):
        min_index = self.point_to_index(box.min_bound)
        max_index = self.point_to_index(box.max_bound)
        clipped_box = Box(self.index_to_point(min_index),
                          self.index_to_point(max_index))
        subgrid = Grid(clipped_box, max_index - min_index + 1)
        indices = tuple(slice(lo, hi+1) for lo, hi in zip(min_index, max_index))
        return subgrid, indices

    def index_to_point(self, index):
        """Convert index in the represented mesh to coordinate point."""
        return self.box.min_bound + index * self.raster

    def point_to_index(self, point):
        """Convert coordinate point to index in the represented mesh."""
        point = np.asarray(point)
        if point.ndim == 2:
            return np.array([self.point_to_index(p) for p in point])
        return array_toint(
            np.clip(np.round((point - self.box.min_bound) / self.raster),
                    self.min_index,
                    self.max_index))

    def xi(self):
        """Generate grid points for interpolation."""
        return np.array(list(
            itertools.product(*(range(int(num)) for num in self.shape))
        )) * (self.box.size / self.shape) + self.box.min_bound


#----------------------------------------
# Distributions
#----------------------------------------

# TODO: smooth ellipses by integrating value in the square

def elliptic_distance(grid, ellipse_center, ellipse_shape):
    # Normalize to unit box:
    ellipse_center = (ellipse_center - grid.box.min_bound) / grid.box.size
    ellipse_shape = ellipse_shape / grid.box.size
    # collect contributions for each dimension
    axis_contributions = (
        ((x-x0)/r) ** 2
        for num, x0, r in zip(grid.shape, ellipse_center, ellipse_shape)
        for x in [np.linspace(0, 1, num)])
    mesh = np.meshgrid(*axis_contributions, indexing='ij')
    return np.sum(mesh, axis=0)


def normal_distribution(grid, ellipse_center, ellipse_shape):
    # Normalize to unit box:
    ellipse_center = (ellipse_center - grid.box.min_bound) / grid.box.size
    ellipse_shape = ellipse_shape / grid.box.size
    # collect contributions for each dimension
    axis_contributions = (
        (x-x0)**2/(2*sigma**2)
        for num, x0, sigma in zip(grid.shape, ellipse_center, ellipse_shape)
        for x in [np.linspace(0, 1, num)])
    # need to specify indexing=ij, to avoid a transposition in the first two
    # arguments:
    mesh = np.meshgrid(*axis_contributions, indexing='ij')
    collected = np.sum(mesh, axis=0)
    # Could alternatively use a solution based on numpy arrays' broadcasting
    # ability::
    #collected = sum_(
    #    contrib.reshape(unit_shape(grid.box.dim, axis, len(contrib)))
    #    for axis, contrib in enumerate(axis_contributions))
    return np.exp(-collected)


def gaussian_filter_global_approximate(grid, points, values, radius):
    result = np.zeros(grid.shape)
    counts = np.zeros(grid.shape, dtype=int)
    for point, value in zip(points, values):
        index = grid.point_to_index(point)
        result[index] += value
        counts[index] += 1
    region = ~np.isclose(counts, 0)
    result[region] /= counts[region]
    return scipy.ndimage.gaussian_filter(result, radius)


def moving_average(grid, points, values, widths, radius, threshold=1):
    def filt(subgrid, point, width):
        dist = np.ones(subgrid.shape)
        ance = elliptic_distance(subgrid, point, width)
        dist[ance > threshold] = 0
        return dist
    return local_linear_filter(grid, points, values, widths, radius, filt)


def gaussian_filter_local_exact(grid, points, values, widths, radius,
                                threshold=1/np.exp(2)):
    def filt(subgrid, point, width):
        dist = normal_distribution(subgrid, point, width)
        dist[dist < threshold] = 0
        return dist
    return local_linear_filter(grid, points, values, widths, radius, filt)


def local_linear_filter(grid, points, values, widths, radius, func):
    points, values, widths = _prep(points, values, widths)
    result = np.zeros(grid.shape)
    counts = np.zeros(grid.shape, dtype=int)
    for point, value, width in zip(points, values, widths):
        subbox = Box(point - radius,
                     point + radius)
        subgrid, indices = grid.subgrid(subbox)
        dist = func(subgrid, point, width*radius)
        result[indices] += dist * value
        counts[indices] += ~np.isclose(dist, 0)
    region = ~np.isclose(counts, 0)
    result[region] /= counts[region]
    return result

#----------------------------------------
# Find regions that are far away from any measured points
#----------------------------------------

def far_points__weighted_cumulative(
        grid, points, values, widths, radius, threshold=1/np.exp(2)):

    points, values, widths = _prep(points, values, widths)

    # place an normal distribution around each point with the radius weighted
    # by the corresponding intensity
    dists = (normal_distribution(grid, point, width*radius) * value
             for point, value, width in zip(points, values, widths))

    # sum up contributions
    zero_mask = sum_(dists) < threshold

    return grid.index_to_point(where(zero_mask))


def far_points__weighted_individual(
        grid, points, values, widths, radius, threshold=1):

    points, values, widths = _prep(points, values, widths)

    # place an ellipse around each point with the radius weighted by the
    # corresponding intensity
    dists = (elliptic_distance(grid, point, width*radius) / value
             for point, value, width in zip(points, values, widths))

    # generate masks for individual points and compute their disjunction
    zero_mask = product_(d > threshold for d in dists)

    return grid.index_to_point(where(zero_mask))


def restrict_to_polytope(facets, candidates):
    # select only those points in the convex hull
    normal, offset = facets[:,:-1], facets[:,-1]
    return [p for p in candidates
            if np.all(np.dot(normal, p.T) + offset <= 0)]
    #return candidates[np.all(normal @ candidates.T + offset <= 0, axis=0)]


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


#----------------------------------------
# Tools for the main
#----------------------------------------

def scatter(grid, points, values, widths, radius):
    """Generate a nice scatter plot."""
    points, values, widths = _prep(points, values, widths)
    # use `sum_(())` instead of `np.sum([])` to avoid huge memory bloat (!)
    return sum_(normal_distribution(grid, point, width*radius) * value
                for point, value, width in zip(points, values, widths))
