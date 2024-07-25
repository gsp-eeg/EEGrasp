"""EEGRasP module.

Contains the class EEGrasp which is used to analyze EEG signals
based graph signal processing.
"""

import numpy as np
from pygsp2 import graphs, learning, graph_learning
from tqdm import tqdm  # TODO: Does it belong here?


class EEGrasp():
    """Class containing functionality to analyze EEG signals.

    Parameters
    ----------
    data : ndarray
        2D or 3D array. Where the first dim are channels and the second is
        samples. If 3D, the first dimension is trials.
    eeg_pos : ndarray
        Position of the electrodes.
    ch_names : ndarray | list
        Channel names.

    Notes
    -----
    Gaussian Kernel functionallity overlapping with PyGSP2 toolbox. This has
    been purposefully added.
    """

    def __init__(self, data=None, coordinates=None, labels=None):
        """
        Parameters
        ----------
        data : ndarray | mne.Evoked | mne.BaseRaw | mne.BaseEpochs | None
            2D array. Where the first dim are channels and the second is
            samples. If 3D, the first dimension is trials. If an mne object is
            passed, the data will be extracted from it along with the
            coordinates and labels of the channels. If `None`, the class will
            be initialized without data. Default is `None`.
        coordinates : ndarray | list | None
            N-dim array or list with position of the electrodes. Dimensions mus
            coincide with the number of channels in `data`. If not provided the
            class instance will not have coordinates associated with the
            nodes. Some functions will not work without this information but
            can be provided later. Default is `None`.
        labels : ndarray | list | None
            Channel names. If not provided the class instance will not have
            labels associated with the nodes. Some functions will not work
            without this information but can be provided later. If `None` then
            the labels will be set to a range of numbers from 0 to the number
            of channels in the data. Default is `None`.
        """

        self.data = data
        self.coordinates = coordinates
        self.labels = labels
        self.distances = None
        self.graph_weights = None
        self.graph = None

    def euc_dist(self, pos):
        """Compute the euclidean distance based on a given set of possitions.

        Parameters
        ----------
        pos : ndarray.
            2d or 3d array of channels by feature dimensions.

        Returns
        -------
        output: ndarray.
            Dimension of the array is number of channels by number of channels
            containing the euclidean distance between each pair of channels.
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

    def compute_distance(self, coordinates=None, method='Euclidean', normalize=True):
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

        Returns
        -------
        distances : ndarray
            Distances to be used for the graph computation.
        """

        # If passed, used the coordinates argument
        if isinstance(coordinates, type(None)):
            coordinates = self.coordinates.copy()

        if method == 'Euclidean':
            distances = self.euc_dist(coordinates)
            np.fill_diagonal(distances, 0)

        if normalize:
            # Normalize distances
            distances = distances - np.amin(distances)
            distances = distances / np.amax(distances)

        self.distances = distances

        return distances

    def compute_graph(self, W=None, epsilon=.5, sigma=.1):
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
            distances = self.distances
            # Check that there is a weight matrix is not a None
            if distances is None:
                raise TypeError(
                    'No distances found. Distances have to be computed if W is not provided')
            graph_weights = self.gaussian_kernel(distances, sigma=sigma)
            graph_weights[distances > epsilon] = 0
            np.fill_diagonal(graph_weights, 0)
            graph = graphs.Graph(graph_weights)
        else:
            graph_weights = W
            graph = graphs.Graph(W)

        if self.coordinates is not None:
            graph.set_coordinates(self.coordinates)

        self.graph = graph
        self.graph_weights = graph_weights

        return graph

    def interpolate_channel(self, missing_idx: int | list[int] | tuple[int], graph=None, data=None):
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

        # Check if values are passed or use the instance's
        if isinstance(data, type(None)):
            data = self.data.copy()
        if isinstance(graph, type(None)):
            graph = self.graph

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

    def _return_results(self, error, signal, vparameter, param_name):
        """Function to wrap results into a dictionary.

        Parameters
        ----------
        error : ndarray
            Errors corresponding to each tried parameter.
        vparameter : ndarray
            Values of the parameter used in the fit function.
        signal : ndarray
            Reconstructed signal.

        Notes
        -----
        In order to keep everyting under the same structure this function
        should be used to return the results of any self.fit_* function.
        """
        best_idx = np.argmin(np.abs(error))
        best_param = vparameter[best_idx]

        results = {'error': error,
                   'signal': signal,
                   f'best_{param_name}': best_param,
                   f'{param_name}': vparameter}

        return results

    def _vectorize_matrix(self, mat):
        """
        Vectorize a simetric matrix using the lower triangle.

        Returns
        -------
        mat : ndarray.
            lower triangle of mat
        """
        tril_indices = np.tril_indices(len(mat), -1)
        vec = mat[tril_indices]

        return vec

    def fit_epsilon(self, missing_idx: int | list[int] | tuple[int], data=None,
                    distances=None, sigma=0.1):
        """Find the best distance to use as threshold.

        Parameters
        ----------
        missing_idx : int
            Index of the missing channel. Not optional.
        data : ndarray | None
            2d array of channels by samples. If None, the function will use the
            data computed in the instance of the class (`self.data`). Default
            is `None`.
        distances : ndarray | None.
            Unthresholded distance matrix (2-dimensional array). It can be
            passed to the instance of the class or as an argument of the
            method. If None, the function will use the distance computed in the
            instance of the class (`self.distances`). Default is `None`.
        sigma : float
            Parameter of the Gaussian Kernel transformation. Default is 0.1.

        Returns
        -------
        results : dict
            Dictionary containing the error, signal, best_epsilon and epsilon
            values.

        Notes
        -----
        It will itterate through all the unique values of the distance matrix.
        data : 2-dimensional array. The first dim. is Channels
        and second is time. It can be passed to the instance class or the method
        """
        # Check if values are passed or use the instance's
        if isinstance(distances, type(None)):
            distances = self.distances.copy()
        if isinstance(data, type(None)):
            data = self.data.copy()

        if isinstance(distances, type(None)) or isinstance(data, type(None)):
            raise TypeError('Check data or W arguments.')

        # Vectorize the distance matrix
        dist_tril = self._vectorize_matrix(distances)

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
            reconstructed = self.interpolate_channel(
                missing_idx=missing_idx, graph=graph, data=signal)
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

        results = self._return_results(error, signal, vdistances, 'epsilon')
        return results

    def fit_sigma(self, missing_idx: int | list[int] | tuple[int], data=None,
                  distances=None, epsilon=0.5, min_sigma=0.1, max_sigma=1.,
                  step=0.1):
        """Find the best parameter for the gaussian kernel.

        Parameters
        ----------
        missing_idx : int | list | tuple
            Index of the missing channel.
        data : ndarray | None
            2d array of channels by samples. If None, the function will use the
            data computed in the instance of the class (`self.data`).
        distances : ndarray | None
            Distance matrix (2-dimensional array). It can be passed to the
            instance of the class or as an argument of the method. If None, the
            function will use the distance computed in the instance of the
            class (`self.distances`).
        epsilon : float
            Maximum distance to threshold the array. Default is 0.5.
        min_sigma : float
            Minimum value for the sigma parameter. Default is 0.1.
        max_sigma : float
            Maximum value for the sigma parameter. Default is 1.
        step : float
            Step for the sigma parameter. Default is 0.1.

        Notes
        -----
        Look for the best parameter of sigma for the gaussian kernel. This is
        done by interpolating a channel and comparing the interpolated data to
        the real data. After finding the parameter the graph is saved and
        computed in the instance class. The distance threshold is maintained.

        """

        # Check if values are passed or use the class instance's
        if isinstance(distances, type(None)):
            distances = self.distances.copy()
        if isinstance(data, type(None)):
            data = self.data.copy()

        if isinstance(distances, type(None)) or isinstance(data, type(None)):
            raise TypeError('Check data or W arguments.')

        # Create array of parameter values
        vsigma = np.arange(min_sigma, max_sigma, step=step)

        # Create time array
        time = np.arange(data.shape[1])

        # Mask to ignore missing channel
        ch_mask = np.ones(data.shape[0]).astype(bool)
        ch_mask[missing_idx] = False

        # Simulate eliminating the missing channel
        signal = data.copy()
        signal[missing_idx, :] = np.nan

        # Allocate array to reconstruct the signal
        all_reconstructed = np.zeros([len(vsigma), len(time)])

        # Allocate Error array
        error = np.zeros([len(vsigma)])

        # Loop to look for the best parameter
        for i, sigma in enumerate(tqdm(vsigma)):

            # Compute thresholded weight matrix
            graph = self.compute_graph(distances, epsilon=epsilon, sigma=sigma)

            # Interpolate signal, iterating over time
            reconstructed = self.interpolate_channel(
                missing_idx=missing_idx, graph=graph, data=signal)
            all_reconstructed[i, :] = reconstructed[missing_idx, :]

            # Calculate error
            error[i] = np.linalg.norm(
                data[missing_idx, :]-all_reconstructed[i, :])

        # Eliminate invalid trials
        valid_idx = ~np.isnan(error)
        error = error[valid_idx]
        vsigma = vsigma[valid_idx]
        all_reconstructed = all_reconstructed[valid_idx, :]

        # Find best reconstruction
        best_idx = np.argmin(np.abs(error))
        best_sigma = vsigma[np.argmin(np.abs(error))]

        # Save best result in the signal array
        signal[missing_idx, :] = all_reconstructed[best_idx, :]

        # Compute the graph with the best result
        graph = self.compute_graph(distances, epsilon=epsilon,
                                   sigma=best_sigma
                                   )

        self.graph = graph

        results = self._return_results(error, signal, vsigma, 'sigma')

        return results

    def learn_graph(self, Z=None, a=0.1, b=0.1,
                    gamma=0.04, maxiter=1000, w_max=np.inf,
                    mode='Average'):
        """Learn the graph based on smooth signals.

        Parameters
        ----------
        Z : ndarray

            Distance between the nodes. If not passed, the function will try to
            compute the euclidean distance using `self.data`. If `self.data` is
            a 2d array it will compute the euclidean distance between the
            channels. If the data is a 3d array it will compute a Z matrix per
            trial, assuming the first dimension in data is
            trials/epochs. Depending on the mode parameter, the function will
            average distance matrizes and learn the graph on the average
            distance or return a collection of adjacency matrices. Default is
            None.
        a : float
            Parameter for the graph learning algorithm, this controls the
            weights of the learned graph. Bigger a -> bigger weights in
            W. Default is 0.1.
        b : float
            Parameter for the graph learning algorithm, this controls the
            density of the learned graph. Bigger b -> more dense W. Default is
            0.1.
        mode : string
            Options are: 'Average', 'Trials'. If 'average', the function
            returns a single W and Z.  If 'Trials' the function returns a
            generator list of Ws and Zs. Default is 'Average'.

        Returns
        -------

        W : ndarray
            Weighted adjacency matrix or matrices depending on mode parameter
            used. If run in 'Trials' mode then Z is a 3d array where the first
            dim corresponds to trials.
        Z : ndarray.
            Used distance matrix or matrices depending on mode parameter
            used. If run in 'Trials' mode then Z is a 3d array where the first
            dim corresponds to trials.
        """

        # If no distance matrix is given compute based on
        # data's euclidean distance
        if Z is None:
            data = self.data.copy()

        # Check if data contains trials
        if data.ndim == 3:

            Zs = np.zeros((data.shape[0], data.shape[1], data.shape[1]))

            # Check if we want to return average or trials
            if mode == 'Trials':

                Ws = np.zeros(
                    (data.shape[0], data.shape[1], data.shape[1]))
                for i, d in enumerate(tqdm(data)):
                    # Compute euclidean distance
                    Z = self.euc_dist(d)

                    W = graph_learning.graph_log_degree(
                        Z, a, b, gamma=gamma, w_max=w_max, maxiter=maxiter)
                    W[W < 1e-5] = 0

                    Ws[i, :, :] = W.copy()
                    Zs[i, :, :] = Z.copy()

                return Ws, Zs

            elif mode == 'Average':

                for i, d in enumerate(tqdm(data)):
                    # Compute euclidean distance
                    Zs[i, :, :] = self.euc_dist(d)

                Z = np.mean(Zs, axis=0)
                W = graph_learning.graph_log_degree(
                    Z, a, b, gamma=gamma, w_max=w_max, maxiter=maxiter)
                W[W < 1e-5] = 0

                return W, Z
        else:
            Z = self.euc_dist(d)

            W = graph_learning.graph_log_degree(
                Z, a, b, gamma=gamma, w_max=w_max, maxiter=maxiter)
            W[W < 1e-5] = 0

            return W, Z
