"""
IO and generic application utilities.
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import contextlib
import time
import sys

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot2d(grid, image):
    image = np.asarray(image).reshape(grid.shape)
    fig = plt.figure()
    ax = plt.subplot(111)
    with trace("  plotting"):
        im = ax.imshow(image, extent=grid.box.lrbt(), cmap="viridis")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, orientation='vertical', cax=cax)
    plt.show()


@contextlib.contextmanager
def trace(message):
    print(message, end='')
    sys.stdout.flush()
    start = time.time()
    try:
        yield None
    finally:
        stop = time.time()
        print("  .. {:.3f}s".format(stop - start))
