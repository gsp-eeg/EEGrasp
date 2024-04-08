""""
creation date: 21/03/2024
autor: jrodino14@gmail.com
description:
Example script to calculate distance from an electrode montage from MNE.
Requirements: None
"""
# %% Load Packages
import mne
import numpy as np
import matplotlib.pyplot as plt
from EEGraSP.eegrasp import EEGraSP

# %%

montage = mne.channels.make_standard_montage('biosemi64')
ch_names = montage.ch_names
EEG_pos = montage.get_positions()['ch_pos']
# Restructure into array
EEG_pos = np.array([pos for _, pos in EEG_pos.items()])

# %% Plot Montage

fig = montage.plot(kind='3d', show=False)
fig.gca().view_init(azim=70, elev=15)  # set view angle for tutorial
plt.title('Electrode Positions in 3d')
plt.show()

# %% Calculate electrode distance

eegrasp = EEGraSP()
W = eegrasp.compute_distance(EEG_pos, method='Euclidean')

# %% Plot distance matrix

im = plt.imshow(W, cmap='gray')
plt.title('Electrode Distance Matrix')
plt.colorbar(label='Euc. Distance [m]')
plt.show()
