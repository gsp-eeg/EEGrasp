r"""Utils.
=====

Utils functions used in EEGrasp.
"""

import numpy as np


def euc_dist(pos):
    """Compute the euclidean distance based on a given set of positions.

    Parameters
    ----------
    pos : ndarray.
        2d or 3d array of channels by feature dimensions.

    Returns:
    -------
    output: ndarray.
        Dimension of the array is number of channels by number of channels
        containing the euclidean distance between each pair of channels.
    """
    from scipy.spatial import distance_matrix

    distance = np.zeros([pos.shape[0], pos.shape[0]],
                        dtype=np.float64)  # Allocate variable
    pos = pos.astype(float)
    distance = distance_matrix(pos, pos)
    return distance


def compute_distance(coordinates=None, method='Euclidean', normalize=True):
    """Computing the distance based on electrode coordinates.

    Parameters
    ----------
    coordinates : ndarray | None
        N-dim array with position of the electrodes. If `None` the class
        instance will use the coordinates passed at initialization. Default
        is `None`.
    method : string
        Options are: 'Euclidean'. Method used to compute the distance matrix.
    normalize : bool
        If True, the distance matrix will be normalized before being
        returned. If False, then the distance matrix will be returned and
        assigned to the class' instance without normalization.

    Returns:
    -------
    distances : ndarray
        Distances to be used for the graph computation.
    """
    # Otherwise use the instance's coordinates
    if method == 'Euclidean':
        distances = euc_dist(coordinates)
        np.fill_diagonal(distances, 0)

    if normalize:
        # Normalize distances
        distances = distances - np.amin(distances)
        distances = distances / np.amax(distances)

    return distances
