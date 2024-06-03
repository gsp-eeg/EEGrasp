"""Using the algorithm proposed in Kalofolias et al. 2019,
and implemented in Pygsp2, learn the graph from EEG signals.
This example follows the methods described in Miri et al.
to run this example you will need to download the data 
described in the paper"""

# %% Import libraries

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.io import loadmat
from eegrasp import EEGraSP

# Set working directory to the file directory
os.chdir(os.path.dirname(__file__))

# Instantiate EEGraSP
gsp = EEGraSP()

# %% Load Electrode montage and dataset
data = loadmat('data/data_set_IVa_aa.mat')
eeg = (data['cnt']).astype(float) * 0.1  # Recomendation: to set to uV
events = np.squeeze(data['mrk'][0, 0][0])
info = data['nfo'][0, 0]
ch_names = [ch_name[0] for ch_name in info[2][0, :]]
FS = info[1][0, 0]
pos = np.array([info[3][:, 0], info[4][:, 0]]) / 10  # Weird behavior from MNE

times = np.array([0.5, 2.5])  # Trial window times
samples = times * FS  # Convert window to samples
s_len = int(np.diff(np.abs(samples))[0])  # Set window length in samples

# %% Preprocessing in MNE

# Create structure
mne_info = mne.create_info(ch_names=ch_names, sfreq=FS, ch_types='eeg')
data = mne.io.RawArray(eeg.T, mne_info)

# Extract events and annotate
mne_events = np.zeros((len(events), 3))
mne_events[:, 0] = events
annotations = mne.annotations_from_events(mne_events, FS)
data = data.set_annotations(annotations)
events2, events_id = mne.events_from_annotations(data)

# Reference data to average
data, ref_data = mne.set_eeg_reference(data, ref_channels='average')

# Filter between 8 and 30 Hz
data = data.filter(l_freq=8, h_freq=30, n_jobs=-1)

# Epoch and Crop epochs
epochs = mne.Epochs(data, events2, tmin=0.0, tmax=2.5,
                    baseline=(0, 0.5), preload=True)
# epochs = epochs.crop(0.5, None)

epochs_data = epochs.get_data()

# %%% Compute the average euclidean distance between the channels
gsp.data = epochs_data
gsp.coordenates = pos
W, Z = gsp.learn_graph(a=0.4, b=0.4)

gsp.compute_graph(W)

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

# %% Plot W

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

# %% Extract eigenvalues and eigenvectors/eigenmodes

G = gsp.graph
G.compute_laplacian()
G.compute_fourier_basis()
eigenvectors = np.array(G.U)
eigenvalues = np.array(G.e)

# %% Plot Eigenvalue index vs eivenvalue
plt.scatter(eigenvalues, np.arange(0, len(eigenvalues)),
            s=50, color='purple')
plt.plot(eigenvalues, np.arange(0, len(eigenvalues)),
         linewidth=3, color='black')
plt.xlabel('Eigenvalue')
plt.ylabel('Eigenvalue Index')
plt.show()

# %% Plot eigenmodes

SCALE = 0.2
vlim = (-np.amax(np.abs(eigenvectors))*SCALE,
        np.amax(np.abs(eigenvectors))*SCALE)


fig, axs = plt.subplots(2, 11, figsize=(14, 4))
for i, ax in enumerate(axs.flatten()):
    im, cn = mne.viz.plot_topomap(eigenvectors[:, i], pos.T,
                                  sensors=True, axes=ax, cmap='RdBu_r',
                                  vlim=vlim, show=False)
    core = r'\u208'
    subscript = [(core+i+'').encode().decode('unicode_escape')
                 for i in str(i+1)]
    subscript = ''.join(subscript)
    ax.text(-0.1, -0.15, r'$\lambda$' + subscript +
            ' = ' + f'{eigenvalues[i]:.3f}')

fig.subplots_adjust(0, 0, 0.85, 1, 0, -0.5)
cbar = fig.add_axes([0.87, 0.1, 0.05, 0.8])
plt.colorbar(im, cax=cbar)
plt.show()
