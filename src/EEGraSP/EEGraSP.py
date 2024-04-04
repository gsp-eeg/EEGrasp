""""
creation date: 21/03/2024
Author: jrodino14@gmail.com
This script defines the main class in the EEGRraSP package.
No inputs are required for the class to initialize, though, for the
computation of graphs and fitting to functional data: EEG_pos,
data and ch_names are required.
"""
import numpy as np
from pygsp import graphs, learning
from tqdm import tqdm


class EEGraSP():
    """"
    Class containing functionality to analyze EEG signals.

    Parameters
    ----------
    data
    eeg_pos
    ch_names

    Notes
    -----
    Gaussian Kernel functionallity overlapping with Pygsp toolbox. This has been purposefully added.
    """

    def __init__(self, data=None, coordenates=None, labels=None):

        self.data = data
        self.coordenates = coordenates
        self.labels = labels
        self.distances = None
        self.graph_weights = None
        self.graph = None

    def euc_dist(self, pos):
        """" 
        Compute the euclidean distance based on a given set of possitions.

        Parameters
        ----------
        input: pos -> 2d or 3d array of channels by dimensions

        Returns
        -------
        output: 2d array of channels by channels with the euclidean distance. 
        description: compute the euclidean distance between every channel in the array 
        """
        distance = np.zeros([pos.shape[0], pos.shape[0]],
                            dtype=np.float64)  # Alocate variable
        pos = pos.astype(float)
        for dim in range(pos.shape[1]):
            # Compute the component corresponding to each dimension. Add it to the array
            distance += np.power(pos[:, dim][:, None]-pos[:, dim][None, :], 2)
        distance = np.sqrt(distance)

        return distance

    def gaussian_kernel(self, x, sigma=0.1):
        """"
        Filter ussing a gaussian kernel.

        Notes
        -----
        This function is supposed to be used in the pygsp module but
        is repeated here since there is an error in the available version
        of the toolbox (03/04/2024 dd/mm/yyyy) 
        """
        return np.exp(-np.power(x, 2.) / (2.*np.power(float(sigma), 2)))

    def compute_distance(self, coordinates=None, method='Euclidean'):
        """
        Method for computing the distance.

        Returns
        -------
        Distances to be used for the graph computation.

        """

        # If passed, used the coordinates argument
        if isinstance(coordinates, type(None)):
            coordinates = self.coordenates.copy()

        if method == 'Euclidean':
            distances = self.euc_dist(coordinates)
            np.fill_diagonal(distances, np.nan)

        self.distances = distances

        return distances

    def compute_graph(self, distances=None, epsilon=.1, sigma=.1):
        """"
        W -> if W is passed, then the graph is computed. 
        Otherwise the graph will be computed with self.W.
        W should correspond to a non-sparse 2-D array.
        Epsilon -> maximum distance to threshold the array.
        sigma -> Sigma parameter for the gaussian kernel.

        method: NN -> Nearest Neighbor
                Gaussian -> Gaussian Kernel used based on the self.W matrix

        output: Graph structure from pygsp        
        """

        # If passed, used the W matrix
        if isinstance(distances, type(None)):
            distances = self.distances.copy()

        # Check that there is a weight matrix is not a None
        if isinstance(distances, type(None)):
            raise TypeError('Weight matrix cannot be None type')

        graph_weights = self.gaussian_kernel(distances, sigma=sigma)
        graph_weights[distances > epsilon] = 0
        np.fill_diagonal(graph_weights, 0)
        graph = graphs.Graph(graph_weights)

        self.graph = graph
        self.graph_weights = graph_weights
        return graph

    def interpolate_channel(self, graph=None, data=None, missing_idx=None):
        """"
        Interpolate missing channel.
        """

        # Check if values are passed or use the instance's
        if isinstance(data, type(None)):
            data = self.data.copy()
        if isinstance(graph, type(None)):
            graph = self.graph

        elif isinstance(missing_idx, type(None)):
            raise TypeError('Parameter missing_idx not specified.')

        time = np.arange(data.shape[1])  # create time array
        mask = np.ones(data.shape[0], dtype=bool)  # Maksing array
        mask[missing_idx] = False

        # Allocate new data array
        reconstructed = np.zeros(data.shape)
        # Iterate over each timepoint
        for t in time:
            reconstructed[:, t] = learning.regression_tikhonov(graph, data[:, t],
                                                               mask, tau=0)
        return reconstructed

    def fit_graph_to_data(self, data=None, distances=None, sigma=0.1,
                          missing_idx=None):
        """"
        Returns the graph that best reconstructs the entire
        segment of data.

        Parameters
        ----------
        distances -> Unthresholded distance matrix (2-dimensional array). 
        It can be passed to the instance of
        the class or as an argument of the method.
        sigma -> parameter of the Gaussian Kernel transformation

        Notes
        -----
        It will itterate through all the unique values of the distance matrix.
        data -> 2-dimensional array. The first dim. is Channels 
        and second time. It can be passed to the instance class or the method


        """
        # Check if values are passed or use the instance's
        if isinstance(distances, type(None)):
            distances = self.distances.copy()
        if isinstance(data, type(None)):
            data = self.data.copy()

        if isinstance(distances, type(None)) or isinstance(data, type(None)):
            raise TypeError('Check data or W arguments.')
        if isinstance(missing_idx, type(None)):
            raise TypeError('Parameter missing_idx not specified.')

        # Vectorize the distance matrix
        tril_indices = np.tril_indices(len(distances), -1)
        dist_tril = distances[tril_indices]

        # Sort and extract unique values
        vdistances = np.sort(np.unique(dist_tril))

        # Create time array
        time = np.arange(data.shape[1])

        # Mask to ignore missing channel
        ch_mask = np.ones(data.shape[0]).astype(bool)
        ch_mask[missing_idx] = False

        # Simulate eliminating the missing channel
        signal = data.copy()
        signal[missing_idx, :] = np.nan

        # Allocate array to reconstruct the signal
        all_reconstructed = np.zeros([len(vdistances), len(time)])

        # Allocate Error array
        error = np.zeros([len(vdistances)])

        # Loop to look for the best parameter
        for i, epsilon in enumerate(tqdm(vdistances)):

            # Compute thresholded weight matrix
            graph = self.compute_graph(distances, epsilon=epsilon, sigma=sigma)

            # Interpolate signal, iterating over time
            reconstructed = self.interpolate_channel(graph, signal,
                                                     missing_idx=missing_idx)
            all_reconstructed[i, :] = reconstructed[missing_idx, :]

            # Calculate error
            error[i] = np.linalg.norm(
                data[missing_idx, :]-all_reconstructed[i, :])

        # Eliminate invalid distances
        valid_idx = ~np.isnan(error)
        error = error[valid_idx]
        vdistances = vdistances[valid_idx]
        all_reconstructed = all_reconstructed[valid_idx, :]

        # Find best reconstruction
        best_idx = np.argmin(np.abs(error))
        best_epsilon = vdistances[np.argmin(np.abs(error))]

        # Save best result in the signal array
        signal[missing_idx, :] = all_reconstructed[best_idx, :]

        # Compute the graph with the best result
        graph = self.compute_graph(distances, epsilon=best_epsilon,
                                   sigma=sigma
                                   )

        results = {'error': error,
                   'signal': signal,
                   'best_epsilon': best_epsilon,
                   'distances': vdistances}

        return results
