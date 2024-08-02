r"""
Interpolate
===========

Contains the functions used in EEGrasp interpolate data
"""

import numpy as np
from pygsp2 import learning


def interpolate_channel(missing_idx: int | list[int] | tuple[int],
                        graph=None,
                        data=None):
    """Interpolate missing channel.
    Parameters
    ----------
    missing_idx : int | list of int | tuple of int
        Index of the missing channel. Not optional.
    graph : PyGSP2 Graph object | None
        Graph to be used to interpolate a missing channel. If None, the
        function will use the graph computed in the instance of the class
        (`self.graph`). Default is None.

    data : ndarray | None
        2d array of channels by samples. If None, the function will use the
        data computed in the instance of the class (`self.data`). Default
        is None.

    Returns
    -------
    reconstructed : ndarray
        Reconstructed signal.
    """

    time = np.arange(data.shape[1])  # create time array
    mask = np.ones(data.shape[0], dtype=bool)  # Maksing array
    mask[missing_idx] = False

    # Allocate new data array
    reconstructed = np.zeros(data.shape)
    # Iterate over each timepoint
    for t in time:
        reconstructed[:, t] = learning.regression_tikhonov(graph,
                                                           data[:, t],
                                                           mask,
                                                           tau=0)
    return reconstructed
