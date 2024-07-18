# %% Load Packages
import mne
import numpy as np
import matplotlib.pyplot as plt
from eegrasp import EEGrasp

# %%

montage = mne.channels.make_standard_montage('biosemi64')
ch_names = montage.ch_names
EEG_pos = montage.get_positions()['ch_pos']
# Restructure into array
EEG_pos = np.array([pos for _, pos in EEG_pos.items()])

# %% Calculate electrode distance

gsp = EEGrasp(coordinates=EEG_pos, labels=ch_names)
Z = gsp.compute_distance(EEG_pos, method='Euclidean')
G = gsp.compute_graph(sigma=0.3, epsilon=0.5)

# %% Plot

gsp.plot_graph(montage='biosemi64')
