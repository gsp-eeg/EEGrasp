# EEGraSP: EEG GRaph Signal Processing

This module is meant to be used as a tool for EEG signal analysis based on graph signal analysis methods. The developement of this toolbox takes place in Gitlab:

https://gitlab.com/gsp8332409/eegrasp

EEGraSP package uses other libraries like pygsp and mne for most of the processing and graph signal analysis.

## Installation

The repository has not been officially released yet. In order to install the python package you can use:

```
pip install -i https://test.pypi.org/simple/ EEGraSP==0.0.1
```

Which will download the package from the testpypi repository (https://test.pypi.org/project/EEGraSP/).

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

````
MISSING_IDX = 5
interpolated = egsp.interpolate_channel(missing_idx=MISSING_IDX)
```

To interpolate a channel of your choice the ````MISSING_IDX``` variable should be changed to the index of the corresponding channel. Remember that python indices start from 0.


## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
