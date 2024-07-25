# %% Import libraries
import matplotlib.pyplot as plt
import mne
from matplotlib import animation
import numpy as np

from eegrasp import EEGrasp
# % matplotlib qt

# %% Load Electrode montage and dataset
subjects = np.arange(1, 10)
runs = [4, 8, 12]

# Download eegbci dataset through MNE
# Comment the following line if already downloaded

raw_fnames = [mne.datasets.eegbci.load_data(s, runs) for s in subjects]
raw_fnames = np.reshape(raw_fnames, -1)
raws = [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
raw = mne.concatenate_raws(raws)
# raw = mne.io.read_raw_edf(data_path[0],preload=True)
mne.datasets.eegbci.standardize(raw)
raw.annotations.rename(dict(T1="left", T2="right"))

montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage)
eeg_pos = np.array(
    [pos for _, pos in raw.get_montage().get_positions()['ch_pos'].items()])
ch_names = montage.ch_names

# %% Filter data and extract events
LOW_FREQ = 1  # Hz
HIGH_FREQ = 30  # Hz
raw.filter(LOW_FREQ, HIGH_FREQ, fir_design='firwin',
           skip_by_annotation='edge')
raw, ref_data = mne.set_eeg_reference(raw)

events, events_id = mne.events_from_annotations(raw)

# %% Epoch data
# Exclude bad channels
TMIN, TMAX = -1.0, 3.0
picks = mne.pick_types(raw.info, meg=False, eeg=True,
                       stim=False, eog=False, exclude="bads")
epochs = mne.Epochs(raw, events, events_id,
                    picks=picks, tmin=TMIN,
                    tmax=TMAX, baseline=(-1, 0),
                    detrend=1)

# %% Compute ERP

left = epochs['left'].average()
right = epochs['right'].average()

# %% Initialize EEGraph class

eegsp = EEGrasp(right, eeg_pos, ch_names)
eegsp.compute_distance()  # Calculate distance between electrodes
weights = eegsp.graph_weights#.copy()

# Plot euclidean distance between electrodes
fig, ax = plt.subplots()
ax.set_title('Electrode Distance')
im = ax.imshow(weights, cmap='Reds')
fig.colorbar(im, label='Euclidean Distance')

# Uncomment to save
# fig.savefig('euc_dist.png')
fig.show()

# %% Binarize W based on the histogram's probability mass

weights = eegsp.graph_weights#.copy()
tril_idx = np.tril_indices(len(weights), -1)
vec_W = weights[tril_idx]
count, bins = np.histogram(vec_W, bins=len(vec_W), density=False)

prob = np.cumsum(count/len(vec_W))
th_W = weights > 0

# Initiate figure
fig, axs = plt.subplots(2, 2, figsize=(10, 9))
axs[0, 0].set_title('Cumulative Distribution Function')
axs[0, 0].set_xlabel('Euc. Distance')
axs[0, 0].set_ylabel('Probability')

lines, = axs[0, 0].plot(np.sort(vec_W), prob, color='black')
dot = axs[0, 0].scatter(np.sort(vec_W)[0], prob[0], c='red')

hist = axs[1, 0].hist(vec_W, bins=20, density=True, color='teal')
vline = axs[1, 0].axvline(np.amin(vec_W), color='purple')
axs[1, 0].set_title('Histogram')
axs[1, 0].set_xlabel('Euc. Distance')
axs[1, 0].set_ylabel('Probability Density')

im = axs[0, 1].imshow(th_W, cmap='gray')
axs[0, 1].set_xlabel('Electrode')
axs[0, 1].set_ylabel('Electrode')
axs[0, 1].set_title('Adjacency Matrix')
cbar = fig.colorbar(im, ax=axs[0, 1], ticks=[0, 1])
cbar.ax.set_yticklabels(['Unconnected', 'Connected'])

fig.tight_layout()

# Define function for animation


def update(frame):
    """"
    Create animation function.
    """

    val = np.sort(vec_W)[frame]
    p = prob[frame]

    th_W = weights <= val  # Keep distances lower than the threshold
    np.fill_diagonal(th_W, 0)  # No self loops

    dot.set_offsets([val, p])
    im.set_data(th_W)
    vline.set_data([[val, val], [0, 1]])

    axs[1, 1].clear()
    G = graphs.Graph(th_W)
    G.set_coordinates()
    G.plot(ax=axs[1, 1])

    return (dot, im)


anim = animation.FuncAnimation(fig, update,
                               frames=np.arange(len(prob))[::8],
                               interval=1, blit=False,
                               cache_frame_data=False)

# Uncomment to save animation
# anim.save('G_thr.gif',fps=30)

# %% Compute graph based on nearest neighbors based on euc. distance

DATA = epochs['left'].get_data()

nchannels = DATA.shape[1]
nsamples = DATA.shape[2]
nepochs = DATA.shape[0]

MISSING_IDX = 5

measures = DATA.copy()
mask = np.ones(len(eeg_pos)).astype(bool)
mask[MISSING_IDX] = False
measures[:, ~mask, :] = np.nan

# %% Graph based on gaussian kernel

v_epsilon = np.arange(0.04, 0.2, 0.01)
for e in v_epsilon:

    G = graphs.NNGraph(eeg_pos, 'radius', rescale=False, epsilon=e)
    weights = G.W.toarray()

    fig, axs = plt.subplots(1, 2, figsize=(7, 4))

    im = axs[0].imshow(weights, 'gray', vmin=0, vmax=1)
    fig.colorbar(im, cmap='jet')

    G.set_coordinates()
    G.plot(ax=axs[1])

# %% Interpolate signal

vk = np.arange(3, 10)
# Allocate error matrix of len(vk) x epochs x timepoints
error = np.zeros([len(vk), measures.shape[0]])
recovery = np.zeros([len(vk), nepochs, nsamples])
for i, k in enumerate(tqdm(vk)):
    # Compute graph from EEG distance
    eegsp.compute_graph(knn=k)

    # Reconstruct every epoch
    for ii, epoch in enumerate(measures):

        # Reconstruct every timepoint
        for iii, t in enumerate(epoch.T):

            # Recover a signal
            recovery[i, ii, iii] = learning.regression_tikhonov(
                eegsp.graph, t, mask, tau=0)[MISSING_IDX]

        error[i, ii] = (np.linalg.norm(
            DATA[ii, MISSING_IDX, :] - recovery[i, ii, :]))

# %% Plot
ii = 1
plt.plot(recovery[4, ii, :])
plt.plot(DATA[ii, MISSING_IDX, :])
plt.show()

# %% Plot error
best_k = vk[np.argmin(error, axis=0)]
plt.plot(vk, error)
plt.show()
