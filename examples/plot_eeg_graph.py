""" Example to plot EEG graph using EEGrasp package. The EEG graph is constructed using the 
electrode positions from the Biosemi 64 channel montage. The graph is plotted in 3D and topoplot
formats.
"""
import matplotlib.pyplot as plt
# %% Load Packages
import mne
import numpy as np

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

fig, ax = gsp.plot()
fig, ax = gsp.plot(kind='3d')

plt.show()
