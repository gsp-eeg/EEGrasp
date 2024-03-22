#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs,learning
from EEGraSP.EEGraSP import EEGraSP
import mne
from scipy.stats import zscore
from tqdm import tqdm
import matplotlib.animation as animation 

#%% Load Electrode montage and dataset
subjects = np.arange(1,10)
runs = [4,8,12]

# Download eegbci dataset through MNE
# Comment the following line if already downloaded

raw_fnames = [mne.datasets.eegbci.load_data(s, runs) for s in subjects]
raw_fnames = np.reshape(raw_fnames,-1)
raws = [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
raw = mne.concatenate_raws(raws)
#raw = mne.io.read_raw_edf(data_path[0],preload=True)
mne.datasets.eegbci.standardize(raw)
raw.annotations.rename(dict(T1="left", T2="right"))


montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage)
EEG_pos = np.array([pos for _,pos in raw.get_montage().get_positions()['ch_pos'].items()])
ch_names = montage.ch_names

# %% Filter data and extract events
l_freq = 1 # Hz
h_freq = 30 # Hz
raw.filter(l_freq,h_freq,fir_design='firwin',skip_by_annotation='edge')
raw,ref_data = mne.set_eeg_reference(raw)

events,events_id = mne.events_from_annotations(raw)


# %% Epoch data
# Exclude bad channels
tmin, tmax = -1.0, 3.0
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
epochs = mne.Epochs(raw,events,events_id,
                    picks=picks,tmin=tmin,
                    tmax=tmax,baseline=(-1,0),
                    detrend=1)

# %%
left = epochs['left'].average()
right = epochs['right'].average()

#fig_left = left.plot(gfp=True)
#fig_right = right.plot(gfp=True)

#fig_left.savefig('left_erp.png')
#fig_right.savefig('rihgt_erp.png')

#plt.show()

# # %% Plot topology maps

# # Times to plot
# times = np.arange(0.2,1.2,.2).round(1)
# window = 0.1 # Window length for averaging
# # Look vor vmin and vmax to share scale
# all_data = np.hstack([left.get_data(),right.get_data()])
# vmin = np.amin(all_data.reshape(-1)) * 1e6
# vmax = np.amax(all_data.reshape(-1)) * 1e6
# vlim = (vmin,vmax)

# fig,axs = plt.subplots(2,len(times)+1,figsize=(8,4))
# fig.suptitle('Mental Imaginery',size=16)
# fontdic = {'size':12}


# # Make plots and titles
# fig.text(.3,.87,'Left Hand',fontdic)
# fig = left.plot_topomap(times,average=window,axes=axs[0,:],
#                         vlim=(vmin,vmax))
# fig.text(.3,.42,'Right Hand',fontdic)
# fig = right.plot_topomap(times,average=window,axes=axs[1,:],
#                             vlim=(vmin,vmax))

# # Change subplots titles
# for ax,t in zip(axs[:,:-1].flatten('F'),times.repeat(2)):
#     ax.set_title(r'{} $\pm {}$ s'.format(t,window),size=9)

# fig.tight_layout()
# #fig.savefig('LR_topoplots.png')
# fig.show()

# %% Initialize EEGraph class
data = right.get_data()
eegsp = EEGraSP(data,EEG_pos,ch_names)
W = eegsp.compute_distance() # Calculate distance between electrodes

# Plot euclidean distance between electrodes
fig,ax = plt.subplots()
ax.set_title('Electrode Distance')
im = ax.imshow(W,cmap='Reds')
fig.colorbar(im,label='Euclidean Distance')

# Uncomment to save
#fig.savefig('euc_dist.png')
plt.show()

# Setup mask
missing_idx = 5
mask = np.ones(len(EEG_pos))
mask[missing_idx] = 0
mask = mask.astype(bool)
# Compute the graph with threshold on the
# distance from
# electrodes
G = eegsp.compute_graph(epsilon=0.1)

recovery = np.zeros(data.shape[1])
signal = data.copy()
signal[missing_idx,:] = np.nan
for t in np.arange(data.shape[1]):
    recovery[t] = learning.regression_tikhonov(G,signal[:,t],
                                mask,tau=0)[missing_idx]

plt.plot(recovery)
plt.plot(data[missing_idx,:])
plt.show()

# %%
W = eegsp.compute_distance(EEG_pos)
# Vectorize the distance matrix
tril_indices = np.tril_indices(len(W),-1)
vec_W = W[tril_indices]

# Sort and extract unique values
distances = np.sort(np.unique(vec_W))
plt.plot(distances)

# %% Fit to data
# This process can take a few minutes

eegsp = EEGraSP(data[:,::8],EEG_pos,ch_names)
W = eegsp.compute_distance()
results = eegsp.fit_graph_to_data(missing_idx=5,
                                  weight_method='Gaussian')
# %% 

tril_indices = np.tril_indices(len(W),-1)
vec_W = W[tril_indices]
error = results['Error']
plt.plot(error)
plt.show()

# %%
W = eegsp.compute_distance()
plt.imshow(W)