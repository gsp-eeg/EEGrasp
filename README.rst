=========================================
EEGraSP: EEG GRaph Signal Processing
=========================================


.. image:: https://img.shields.io/readthedocs/eegrasp
   :target: https://eegrasp.readthedocs.io
   :alt: Read the Docs

.. image:: https://img.shields.io/pypi/v/eegrasp
   :target: https://pypi.org/project/eegrasp
   :alt: PyPI - Version
   
.. image:: https://img.shields.io/conda/v/conda-forge/eegrasp
   :target: https://anaconda.org/conda-forge/eegrasp
   :alt: Conda Version
   
.. image:: https://img.shields.io/github/license/gsp-eeg/EEGraSP
   :target: https://github.com/gsp-eeg/eegrasp/blob/main/LICENSE
   :alt: GitHub License

.. image:: https://img.shields.io/pypi/pyversions/eegrasp
   :target: https://pypi.org/project/eegrasp
   :alt: PyPI - Python Version

.. image:: https://img.shields.io/github/actions/workflow/status/gsp-eeg/EEGraSP/release-pypi.yml?label=release
   :target: https://github.com/gsp-eeg/EEGraSP/actions
   :alt: GitHub Actions Workflow Release Status

.. image:: https://img.shields.io/github/actions/workflow/status/gsp-eeg/EEGraSP/test-merge.yml?label=merge
   :target: https://github.com/gsp-eeg/EEGraSP/actions
   :alt: GitHub Actions Workflow Testing Status

.. image:: https://img.shields.io/github/actions/workflow/status/gsp-eeg/EEGraSP/test-main.yml?label=main
   :target: https://github.com/gsp-eeg/EEGraSP/actions
   :alt: GitHub Actions Workflow Testing Status



This module is meant to be used as a tool for EEG signal analysis based on graph signal analysis methods. The development of this toolbox takes place in `GitHub <https://github.com/gsp-eeg/EEGraSP>`_.

EEGraSP package uses other libraries like PyGSP2 and mne for most of the processing and graph signal analysis.

Installation with pip (User Installation)
-----------------------------------------

The EEGraSP is available on PyPI::

     $ pip install eegrasp

Installation with conda (User Installation)
-------------------------------------------

The EEGraSP is available on Conda Forge::

     $ conda install conda-forge::eegrasp

Installation from source (User Installation)
--------------------------------------------

1. Clone the EEGraSP repository into a local directory with git: ``git clone https://github.com/gsp-eeg/eegrasp``
2. Change the current directory to the directory of the downloaded repository. ``cd eegrasp``
3. Install the cloned repository in your preferred Python environment through git. Use: ``pip install -e .``.

Now you are ready to contribute!


Usage
-----

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
-------
MIT licence

Project status
--------------
Still in development.

Acknowledgments
---------------
EEGraSP has been partly funded by FONDECYT REGULAR 1231132 grant, ANILLO ACT210053, and BASAL FB0008 grant.
