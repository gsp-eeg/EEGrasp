"""Using the algorithm proposed in Kalofolias et al. 2019,
and implemented in Pygsp2, learn the graph from EEG signals. This example follows
the methods described in Miri et al."""

# %% Import libraries

import numpy as np
import matplotlib.pyplot as plt
import mne
from pygsp import graph_learning
from eegrasp import EEGraSP

# %% Load Electrode montage and dataset
subjects = np.arange(1, 15)
runs = [4, 8, 12]

# Download eegbci dataset through MNE
# Comment the following line if already downloaded

raw_fnames = [mne.datasets.eegbci.load_data(
    s, runs, path='datasets') for s in subjects]
raw_fnames = np.reshape(raw_fnames, -1)
raws = [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
raw = mne.concatenate_raws(raws)
mne.datasets.eegbci.standardize(raw)
raw.annotations.rename(dict(T1="left", T2="right"))


montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage)
eeg_pos = np.array(
    [pos for _, pos in raw.get_montage().get_positions()['ch_pos'].items()])
ch_names = montage.ch_names

# %% Filter data and extract events
L_FREQ = 8  # Hz
H_FREQ = 30  # Hz
raw.filter(L_FREQ, H_FREQ, fir_design='firwin', skip_by_annotation='edge')
raw, ref_data = mne.set_eeg_reference(raw)

events, events_id = mne.events_from_annotations(raw)

# %% Epoch data
# Exclude bad channels
TMIN, TMAX = (0, 1)
picks = mne.pick_types(raw.info, meg=False, eeg=True,
                       stim=False, eog=False, exclude="bads")
epochs = mne.Epochs(raw, events, events_id,
                    picks=picks, tmin=TMIN,
                    tmax=TMAX, detrend=1, baseline=(0, 0.5))

# %%
left = epochs['left']
erp_left = left.average()

right = epochs['right']
erp_right = right.average()

# %% Instantiate EEGraSP and load trials
eegsp = EEGraSP()
trials = left.get_data()

# %% Compute distance
Zs = np.zeros([trials.shape[0], len(eeg_pos), len(eeg_pos)])
for i, trial in enumerate(trials):
    Zs[i] = eegsp.euc_dist(trial)

# %% Learn graph

Z = np.mean(Zs, axis=0)
a, b = (0.1, 0.1)
W = graph_learning.graph_log_degree(Z, a, b, w_max=1, gamma=0.03, maxiter=1000)
W[W < 1e-5] = 0

plt.figure(figsize=(10, 3))
plt.subplot(121)
plt.imshow(Z, cmap='hot')
plt.colorbar(label='Distance')
plt.subplot(122)
plt.imshow(W, cmap='hot')
plt.colorbar(label='Weight')

plt.tight_layout()
