=========================================
EEGraSP: EEG GRaph Signal Processing
=========================================

.. image:: https://readthedocs.org/projects/eegrasp/badge/?version=latest
   :target: https://eegrasp.readthedocs.io
   :alt: Read the Docs

.. image:: https://img.shields.io/pypi/v/eegrasp.svg
   :target:https://pypi.org/project/eegrasp
   :alt: PyPi

.. image:: https://anaconda.org/conda-forge/eegrasp/badges/version.svg)
   :target: https://anaconda.org/conda-forge/eegrasp
   :alt: Conda Forge

.. image:: https://img.shields.io/pypi/l/eegrasp.svg) 
   :target: https://github.com/gsp-eeg/eegrasp/blob/main/LICENSE
   :alt: Licence

.. image:: https://img.shields.io/pypi/pyversions/eegrasp.svg
   :target: https://pypi.org/project/eegrasp
   :alt: Python Version


This module is meant to be used as a tool for EEG signal analysis based on graph signal analysis methods. The developement of this toolbox takes place in `GitHub <https://github.com/gsp-eeg/EEGraSP>`_

EEGraSP package uses other libraries like PyGSP2 and mne for most of the processing and graph signal analysis.

Installation with pip (User Installation)
------------

The EEGraSP is available on PyPI::

     $ pip install eegrasp

Installation with conda (User Installation)
------------

The EEGraSP is available on Conda Forge::

     $ conda install conda-forge::eegrasp

Installation from source (User Installation)
------------

1. Clone the EEGraSP repository into a local directory with git: ``git clone https://github.com/gsp-eeg/eegrasp``
2. Change the current directory to the directory of the downloaded repository. ``cd eegrasp``
3. Install the cloned repository in your prefered Python enviorment through git. Use: ``pip install -e .``.

Now you are ready to contribute!


Usage
------------

Examples are provided in the `examples <https://github.com/gsp-eeg/EEGraSP/tree/main/examples>`_ folder of the repository:



* The ``electrode_distance.py`` script computes the electrode distance from the standard biosemi64 montage provided in the MNE package.

* The ``ERP_reconstruction.py`` script computes an example ERP from a database provided by MNE. Then, one of the channels is eliminated and reconstructed through Tikhonov Regression. 

Basic steps for the package ussage are:

1. Load the Package

>>> from EEGraSP.eegrasp import EEGraSP

2. Initialize the EEGraSP class instance.

>>> eegsp = EEGraSP(data, eeg_pos, ch_names)

Where:
``data`` is a 2-dimensional numpy array with first dimension being channels and second dimension being the samples of the data. The missing channel should be included with np.nan as each sample.
``eeg_pos`` is a 2-dimensional numpy array with the position of the electrodes. This can be obtained through the MNE library. See examples for more information about how to do this.
``ch_names`` is a list of names for each channel. 

3. Compute the graph based on the electrodes distance. The parameters used to compute the graph need to be provided or estimated. In this case we will provide the parameters epsilon and sigma. To see how to find the best parameter for your data see ``ERP_reconstruction.py`` in the examples folder.


>>> distances = eegsp.compute_distance()
>>> graph_weights = eegsp.compute_graph(epsilon=0.5,sigma=0.1)

4. Interpolate the missing channel.

>>> MISSING_IDX = 5
>>> interpolated = egsp.interpolate_channel(missing_idx=MISSING_IDX)

To interpolate a channel of your choice the ``MISSING_IDX`` variable should be changed to the index of the corresponding channel. Remember that python indices start from 0.

License
------------
MIT licence

Project status
------------
Still in developement.

Acknowledgments
------------
EEGraSP has been partly funded by FONDECYT REGULAR 1231132 grant, ANILLO ACT210053, and BASAL FB0008 grant.
