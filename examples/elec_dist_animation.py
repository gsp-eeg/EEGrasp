# %% Import libraries
import matplotlib.pyplot as plt
import mne
from matplotlib import animation
import numpy as np
from pygsp import graphs
from EEGraSP.eegrasp import EEGraSP
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

eegsp = EEGraSP(right, eeg_pos, ch_names)
eegsp.compute_distance()  # Calculate distance between electrodes
distances = eegsp.distances

# Plot euclidean distance between electrodes
fig, ax = plt.subplots()
ax.set_title('Electrode Distance')
im = ax.imshow(distances, cmap='Reds')
fig.colorbar(im, label='Euclidean Distance')

# Uncomment to save
# fig.savefig('euc_dist.png')
fig.show()

# %% Binarize W based on the histogram's probability mass

tril_idx = np.tril_indices(len(distances), -1)
vec_W = distances[tril_idx]
count, bins = np.histogram(vec_W, bins=len(vec_W), density=False)

prob = np.cumsum(count/len(vec_W))
th_W = distances > 0

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

    epsilon = distances <= val  # Keep distances lower than the threshold
    np.fill_diagonal(epsilon, 0)  # No self loops

    dot.set_offsets([val, p])
    im.set_data(epsilon)
    vline.set_data([[val, val], [0, 1]])

    axs[1, 1].clear()
    graph = graphs.Graph(epsilon)
    graph.set_coordinates()
    graph.plot(ax=axs[1, 1])

    return (dot, im)


anim = animation.FuncAnimation(fig, update,
                               frames=np.arange(len(prob))[::16],
                               interval=.1, blit=False,
                               cache_frame_data=False)

plt.tight_layout()
plt.show()


# Uncomment to save animation
# anim.save('G_thr.gif',fps=30)
