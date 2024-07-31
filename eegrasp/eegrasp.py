r"""
EEGRasP module
==============

Contains the class EEGrasp which is used to analyze EEG signals
based graph signal processing.
"""

import numpy as np
from pygsp2 import graphs, learning, graph_learning
from tqdm import tqdm
from scipy import spatial
import mne
from .viz import plot_graph



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

        # Detect if data is a mne object
        if self._validate_MNE(data):
            self._init_from_mne(data)
        else:
            self.data = data
            self.coordinates = coordinates
            self.labels = labels
        self.distances = None
        self.graph_weights = None
        self.graph = None

    def _init_from_mne(self, data):
        """Initialize EEGrasp attributes from the MNE object.

        Parameters
        ----------
        data : any
            Object to be checked if it is an instance of the valid MNE objects
            allowed by the EEGrasp toolbox.
        """
        info = data.info
        self.data = data.get_data()
        self.coordinates = np.array(
            [pos for _, pos in info.get_montage().get_positions()['ch_pos'].items()])
        self.labels = info.ch_names

    def _validate_MNE(self, data):
        """Check if the data passed is a MNE object and extract the data and
        coordinates.

        Parameters
        ----------
        data : any

            Object to be checked if it is an instance of the valid MNE objects
            allowed by the EEGrasp toolbox.
        """
        is_mne = False
        if isinstance(data, (mne.Epochs, mne.Evoked, mne.io.Raw)):
            is_mne = True

        return is_mne


    def euc_dist(self, pos):
        r"""Docstring overloaded at import time."""
        from .utils import euc_dist
        return euc_dist(pos)


    def gaussian_kernel(self, x, sigma=0.1):
        r"""Docstring overloaded at import time."""
        from .graph_creation import gaussian_kernel
        return gaussian_kernel(x, sigma=0.1)
        

    def compute_distance(self, coordinates=None, method='Euclidean', normalize=True):
        r"""Docstring overloaded at import time."""
        from .utils import compute_distance
        self.distances = compute_distance(coordinates=coordinates, method=method, normalize=normalize, coord0=self.coordinates)
        return self.distances
        

    def compute_graph(self, W=None, epsilon=.5, sigma=.1, distances=None, graph=None, coordinates=None):
        r"""Docstring overloaded at import time."""
        from .graph_creation import compute_graph
        self.graph, self.graph_weights = compute_graph(W=W, epsilon=epsilon, sigma=sigma, distances=distances, graph=graph, coordinates=coordinates, distances0=self.distances)
        return self.graph
        

    def interpolate_channel(self, missing_idx: int | list[int] | tuple[int], graph=None, data=None):
        r"""Docstring overloaded at import time."""
        from .interpolate import interpolate_channel
        return interpolate_channel(missing_idx, graph=graph, data=data, graph0=self.graph, data0=self.data)


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
            graph = self.compute_graph(epsilon=epsilon, sigma=sigma)

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
            Z = self.euc_dist(data)

            W = graph_learning.graph_log_degree(
                Z, a, b, gamma=gamma, w_max=w_max, maxiter=maxiter)
            W[W < 1e-5] = 0

            return W, Z

    def plot(self, graph=None, signal=None, coordinates=None, labels=None, montage=None,
             colorbar=True, axis=None, clabel='Edge Weights', kind='topoplot', show_names=True, **kwargs):
        """ Plot graph over the eeg montage.
        %(eegrasp.viz.plot_graph)s
        """
        return plot_graph(eegrasp=self, graph=graph, signal=signal, coordinates=coordinates, labels=labels, montage=montage,
                          colorbar=colorbar, axis=axis, clabel=clabel, kind=kind, show_names=show_names, **kwargs)
