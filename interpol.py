"""
Utilities for working with numpy arrays and interpolation.
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

import itertools
import functools
import operator
import bisect

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


def hstack(*args):
    return np.hstack(args)


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

    def lrbt(self):
        return (self.min_bound[1],
                self.max_bound[1],
                self.min_bound[0],
                self.max_bound[0])


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
        self.raster = box.size / (self.shape - 1)
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


def slices(array, index):
    for i, v in enumerate(index):
        along_i_axis = list(index)
        along_i_axis[i] = slice(None)
        yield array[along_i_axis], v


def jitter(pdist, x_discrete):
    x = [-1, 0, +1]
    return [
        v + random_linear_interpolate_interval(x, p_nearest, -0.5, +0.5)
        for pdist_i, v in slices(pdist, x_discrete)
        # NOTE: extrapolating (!!) via boundary condition `p=0`:
        for p_nearest in [hstack(0, pdist_i, 0)[v:v+3]]
    ]


def random_linear_interpolate_interval(x, y, x_a, x_b):
    """
    Return a random value in the range `[x_a, x_b)` based on linear
    interpolation of the probabilities `y` at locations `x`. `x` must be
    sorted.
    """
    m = lambda i: (y[i+1]-y[i])/(x[i+1]-x[i])
    i_a = bisect.bisect_right(x, x_a)
    i_b = bisect.bisect_left(x, x_b, i_a)
    p_a = m(i_a-1) * (x_a - x[i_a-1]) + y[i_a-1]
    p_b = m(i_b-1) * (x_b - x[i_b-1]) + y[i_b-1]
    x = hstack(x_a, x[i_a:i_b], x_b)
    y = hstack(p_a, y[i_a:i_b], p_b)
    return random_linear_interpolate(x, y)


def random_linear_interpolate(x, y):
    """
    Sample a random number from the interval [x0, xN) by interpolating
    linearly between N supporting points (x, y) of the probability density
    function. The probability does not need to be normalized.
    """
    x = np.array(x)
    y = np.array(y)
    # probabability for the random number to be in the i'th interval:
    p_i = (y[1:] + y[:-1]) * (x[1:] - x[:-1]) / 2
    # cumulative probability for the random number to be in the i'th interval
    cum_i = hstack(0, np.cumsum(p_i))
    cum_i_norm = cum_i[-1]
    # choose a position within our total probability weight
    p_integral = np.random.uniform(0, cum_i_norm)
    # find the interval
    i = np.searchsorted(cum_i, p_integral, 'right') - 1
    # coordinates of the selected interval
    x0, x1 = x[i:i+2]
    y0, y1 = y[i:i+2]
    # linearly interpolate the 1D probability density function (pdf) in the
    # interval [i, i+1), from now on, we assume that
    pdf_i = [(y1-y0)/(x1-x0), y0]
    # integrate to get the cumulative distribution function (cdf):
    cdf_i = np.polyint(pdf_i)
    cdf_i_norm = np.polyval(cdf_i, x1-x0)
    # position within interval
    p_integral_i = (p_integral - cum_i[i]) / p_i[i] * cdf_i_norm
    # (c=0, but we keep it anyway)
    a, b, c = cdf_i
    if np.isclose(a, 0):
        return x0 + (p_integral_i - c) / b
    return x0 + (-b + (b**2 - 4*a*(c-p_integral_i))**0.5) / (2*a)


#----------------------------------------
# Tools for the main
#----------------------------------------

def scatter(grid, points, values, widths, radius):
    """Generate a nice scatter plot."""
    points, values, widths = _prep(points, values, widths)
    # use `sum_(())` instead of `np.sum([])` to avoid huge memory bloat (!)
    return sum_(normal_distribution(grid, point, width*radius) * value
                for point, value, width in zip(points, values, widths))
