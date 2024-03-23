#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
from EEGraSP.EEGraSP import EEGraSP
import mne 

#%% Load Electrode montage and dataset
subjects = np.arange(1,10)
runs = [4,8,12]

# Download eegbci dataset through MNE
# Comment the following line if already downloaded

raw_fnames = [mne.datasets.eegbci.load_data(s,runs,path='datasets') for s in subjects]
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

# Use only data on the Left condition to find 
# the best distance (epsilon) value
data = left.get_data()

# %% Fit to data
# This process can take a few (~7) minutes

# 1. Define index of the missing channel
missing_idx = 5
# 2. Initialize instance of EEGraSP
eegsp = EEGraSP(data,EEG_pos,ch_names)
# 3. Compute the electrode distance matrix
W = eegsp.compute_distance()
# 4. Find the best parameter for the channel
results = eegsp.fit_graph_to_data(missing_idx=missing_idx,
                                  weight_method='Gaussian')

# %% Plot error graph and results of the interpolation

tril_indices = np.tril_indices(len(W),-1)
vec_W = np.unique(np.sort(W[tril_indices]))
error = results['Error']
best_idx = np.argmin(error[~np.isnan(error)])
signal = results['Signal'][best_idx,:]
best_epsilon = results['best_epsilon']
distances = results['Distances']

plt.subplot(211)
plt.plot(distances,error,color='black')
plt.scatter(distances,error,color='teal',marker='x',
            alpha=0.2)
plt.scatter(best_epsilon,
            error[distances==best_epsilon],
            color='red')
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'RMSE [$V$]')
plt.title('Error')

plt.subplot(212)
plt.title('Reconstructed vs True EEG channel')
plt.plot(signal)
plt.plot(data[missing_idx,:])
plt.xlabel('Time')
plt.ylabel('V')

plt.tight_layout()
plt.show()

# %% Interpolate right ERP based on the left channel
new_data = right.get_data()
# Delete information from the missing channel
new_data[missing_idx,:] = np.nan

# Interpolate channel
interpolated = eegsp.interpolate_channel(data=new_data,
                          missing_idx=missing_idx)

# %% Plot Interpolated Channel
original = right.get_data()
plt.plot(interpolated[missing_idx,:])
plt.plot(original[missing_idx,:])
plt.xlabel('Samples')
plt.ylabel('Voltage')
