"""Using the algorithm proposed in Kalofolias et al. 2019,
and implemented in Pygsp2, learn the graph from EEG signals.
This example follows the methods described in Miri et al."""

# %% Import libraries

import numpy as np
import matplotlib.pyplot as plt
import mne
from pygsp import graph_learning, graphs
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
from eegrasp import EEGraSP

plt.switch_backend('qtagg')
gsp = EEGraSP()

# %% Load Electrode montage and dataset
data = loadmat('data/data_set_IVa_aa.mat')
eeg = (data['cnt']).astype(float) * 0.1  # Recomendation: to set to uV
events = np.squeeze(data['mrk'][0, 0][0])
info = data['nfo'][0, 0]
FS = info[1][0, 0]
pos = np.array([info[3][:, 0], info[4][:, 0]]) / 10

times = np.array([0.5, 2.5])  # Trial window times
samples = times * FS  # Convert window to samples
s_len = int(np.diff(np.abs(samples))[0])  # Set window length in samples

# %% Filter signal
sos = butter(N=3, Wn=[8, 30], fs=FS, btype='bandpass', output='sos')
filt_eeg = sosfiltfilt(sos, eeg, axis=0)

# %% Create matrix with erps
trials = np.zeros((len(events), s_len, eeg.shape[1]))
for i, event in enumerate(events):
    t_idx = (samples+event).astype(int)
    trials[i, :, :] = filt_eeg[t_idx[0]:t_idx[1], :]

Z = []
for trial in trials:
    Z.append(gsp.euc_dist(trial.T))
Z = np.array(Z)
Z = np.mean(Z, axis=0)

tril_idx = np.tril_indices(len(Z), -1)

# %% Plot Z
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.imshow(Z, cmap='hot')
plt.colorbar(label='Distance [uV]')
plt.title('Distance Matrix, Z')


plt.subplot(122)
plt.hist(Z[tril_idx], 10)
plt.xlabel('Distance')
plt.ylabel('N° Count')
plt.title('Histogram')

plt.tight_layout()
plt.show()

# %% Learn Graph Weights

W = graph_learning.graph_log_degree(Z, 0.3, 0.5,
                                    maxiter=1000,
                                    gamma=0.01,
                                    w_max=np.inf)
W[W < 1e-5] = 0

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.imshow(W, cmap='hot')
plt.colorbar(label='Weights')
plt.title('Adjacency Matrix, W')

plt.subplot(122)
plt.hist(W[tril_idx], bins=10, log=True)
plt.xlabel('Distance')
plt.ylabel('N° Count')
plt.title('Histogram')

plt.tight_layout()
plt.show()

# %% Create graph and compute fourier basis

G = graphs.Graph(W, coords=pos.T)
G.compute_laplacian()
G.compute_fourier_basis()
eigenvectors = np.array(G.U)
eigenvalues = np.array(G.e)

# %%
plt.scatter(eigenvalues, np.arange(0, len(eigenvalues)), color='purple')
plt.plot(eigenvalues, np.arange(0, len(eigenvalues)),
         linewidth=2, color='black')
plt.xlabel('Eigenvalue')
plt.ylabel('Eigenvalue Index')

# %%
scale = 0.2
vlim = (-np.amax(np.abs(eigenvectors))*scale,
        np.amax(np.abs(eigenvectors))*scale)

fig, axs = plt.subplots(2, 11, figsize=(14, 4))
for i, ax in enumerate(axs.flatten()):
    im, cn = mne.viz.plot_topomap(eigenvectors[:, i], pos.T,
                                  sensors=True, axes=ax, cmap='RdBu_r',
                                  vlim=vlim, show=True)
    ax.text(-0.1, -0.15, r'$\lambda_{} = $'.format(i+1) +
            f'{eigenvalues[i]:.3f}')

# plt.tight_layout()

fig.subplots_adjust(0, 0, 0.85, 1, 0, -0.5)
cbar = fig.add_axes([0.87, 0.1, 0.05, 0.8])
plt.colorbar(im, cax=cbar)
