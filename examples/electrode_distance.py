# %% Load Packages
import matplotlib.pyplot
import mne
from EEGraSP.EEGraSP import EEGraSP
import numpy as np
import matplotlib.pyplot as plt

# %%

montage = mne.channels.make_standard_montage('biosemi64')
ch_names = montage.ch_names
EEG_pos = montage.get_positions()['ch_pos']
# Restructure into array
EEG_pos = np.array([pos for _,pos in EEG_pos.items()])

# %% Plot Montage

fig = montage.plot(kind='3d',show=False)
fig = fig.gca().view_init(azim=70, elev=15)  # set view angle for tutorial

# %% Calculate electrode distance

eegrasp = EEGraSP()
W = eegrasp.compute_distance(EEG_pos,method='Euclidean')

# %% Plot distance matrix

plt.imshow(W,cmap='gray')
plt.colorbar(label='Euc. Distance [m]')
