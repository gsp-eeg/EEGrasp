# EEGraSP: EEG GRaph Signal Processing

This module is meant to be used as a tool for EEG signal analysis based on graph signal analysis methods. The developement of this toolbox takes place in Gitlab:

https://gitlab.com/gsp8332409/eegrasp

EEGraSP package uses other libraries like pygsp and mne for most of the processing and graph signal analysis.

## Installation with pip (User Installation)

The repository has not been officially released yet. In order to install the python package you can use:

```
pip install git+https://github.com/epfl-lts2/pygsp.git
pip install --extra-index-url https://test.pypi.org/simple/ EEGraSP==0.0.2
```

Which will download the package from the testpypi repository (https://test.pypi.org/project/EEGraSP/). Also, since a lot of the used functions come from the PyGSP it is important to install 
from the PyGSP github repository to avoid version issues.

## Installation from source (Developers and Contribuitors)

You may want to contribute to the project or build functions on top of what we've built here. This installation will always be the most updated version but could also contain some errors or bugs. To install from the repository first you'll have to install the Pygsp fork that we have modified, follow the steps below:

1. Clone the Pygsp fork we've made into a local directory with git: ```git clone https://gitlab.com/gsp8332409/pygsp.git```
2. Change the current directory to the directory of the downloaded repository. ```cd pygsp```
3. Install the cloned repository in your prefered Python enviorment through pip. Use: ```pip install -e .```. If you want the static version of this installation, and are not planning on making changes to the pygsp toolbox, you can drop the "-e" option.

**Now you are ready to install EEGraSP**. Follow the same steps but with the EEGRaSP repository (SUGGESTION: don't clone the repository inside the pygsp repository):

1. Clone the EEGraSP repository into a local directory with git: ```git clone https://gitlab.com/gsp8332409/eegrasp.git```
2. Change the current directory to the directory of the downloaded repository. ```cd eegrasp```
3. Install the cloned repository in your prefered Python enviorment through git. Use: ```pip install -e .```.

Now you are ready to contribute!


## Usage

Examples are provided in the examples folder of the repository:

https://gitlab.com/gsp8332409/eegrasp/-/tree/main/examples?ref_type=heads

* The ```electrode_distance.py``` script computes the electrode distance from the standard biosemi64 montage provided in the MNE package.

* The ```ERP_reconstruction.py``` script computes an example ERP from a database provided by MNE. Then, one of the channels is eliminated and reconstructed through Tikhonov Regression. 

Basic steps for the package ussage are:

1. Load the Package

```
from EEGraSP.eegrasp import EEGraSP
```

2. Initialize the EEGraSP class instance.

```
eegsp = EEGraSP(data, eeg_pos, ch_names)
```

Where:
```data``` is a 2-dimensional numpy array with first dimension being channels and second dimension being the samples of the data. The missing channel should be included with np.nan as each sample.
```eeg_pos``` is a 2-dimensional numpy array with the position of the electrodes. This can be obtained through the MNE library. See examples for more information about how to do this.
```ch_names``` is a list of names for each channel. 

3. Compute the graph based on the electrodes distance. The parameters used to compute the graph need to be provided or estimated. In this case we will provide the parameters epsilon and sigma. To see how to find the best parameter for your data see ```ERP_reconstruction.py``` in the examples folder.

```
distances = eegsp.compute_distance()
graph_weights = eegsp.compute_graph(epsilon=0.5,sigma=0.1)
```

4. Interpolate the missing channel.

```
MISSING_IDX = 5
interpolated = egsp.interpolate_channel(missing_idx=MISSING_IDX)
```

To interpolate a channel of your choice the ```MISSING_IDX``` variable should be changed to the index of the corresponding channel. Remember that python indices start from 0.

## License
MIT licence

## Project status
Still in developement.

## Acknowledgments

EEGraSP has been partly funded by FONDECYT REGULAR 1231132 grant, ANILLO ACT210053, and BASAL FB0008 grant.
