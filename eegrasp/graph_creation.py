r"""
Graph Creation
==============

Contains the functions used in EEGrasp to create Graphs 
"""


import numpy as np
from pygsp2 import graphs

def gaussian_kernel(x, sigma=0.1):
    """Gaussian Kernel Weighting function.

    Notes
    -----
    This function is supposed to be used in the PyGSP2 module but is
    repeated here since there is an error in the available version of the
    toolbox.

    References
    ----------
    * D. I. Shuman, S. K. Narang, P. Frossard, A. Ortega and
    P. Vandergheynst, "The emerging field of signal processing on graphs:
    Extending high-dimensional data analysis to networks and other
    irregular domains," in IEEE Signal Processing Magazine, vol. 30, no. 3,
    pp. 83-98, May 2013, doi: 10.1109/MSP.2012.2235192.
    """
    return np.exp(-np.power(x, 2.) / (2.*np.power(float(sigma), 2)))


def compute_graph(W=None, epsilon=.5, sigma=.1, distances=None, graph=None, coordinates=None, distances0=None):

    """Parameters
    ----------
    W : numpy ndarray | None
        If W is passed, then the graph is computed. Otherwise the graph
        will be computed with `self.W`. `W` should correspond to a
        non-sparse 2-D array. If None, the function will use the distance
        matrix computed in the instance of the class (`self.W`).
    epsilon : float
        Any distance greater than epsilon will be set to zero on the
        adjacency matrix.
    sigma : float
        Sigma parameter for the gaussian kernel.
    method: string
        Options are: "NN" or "Gaussian". Nearest Neighbor or Gaussian
        Kernel used based on the `self.W` matrix respectively depending on
        the method used.

    Returns
    -------
    G: PyGSP2 Graph object.
    """

    # If passed, used the W matrix
    if W is None:
        distances = distances0
        # Check that there is a weight matrix is not a None
        if distances is None:
            raise TypeError(
                'No distances found. Distances have to be computed if W is not provided')
        graph_weights = gaussian_kernel(distances, sigma=sigma)
        graph_weights[distances > epsilon] = 0
        np.fill_diagonal(graph_weights, 0)
        graph = graphs.Graph(graph_weights)
    else:
        graph_weights = W
        graph = graphs.Graph(W)

    if coordinates is not None:
        graph.set_coordinates(coordinates)

    return graph, graph_weights