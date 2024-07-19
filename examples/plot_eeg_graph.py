""" Example to plot EEG graph using EEGrasp package. The EEG graph is constructed using the 
electrode positions from the Biosemi 64 channel montage. The graph is plotted in 3D and topoplot
formats.
"""
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
Z = gsp.compute_distance(EEG_pos)
G = gsp.compute_graph(sigma=0.1, epsilon=0.2)

# %% Plot
info = mne.create_info(ch_names, sfreq=256, ch_types="eeg")
info.set_montage(montage, on_missing="ignore")

# fig, ax = gsp.plot_graph(kind='3d', montage='biosemi64')
fig, ax = gsp.plot_graph(kind='topoplot', montage='biosemi64', cmap='viridis')

plt.show()
