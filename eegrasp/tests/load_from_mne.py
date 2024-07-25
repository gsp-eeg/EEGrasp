"""Use mne epochs to import data and calculate the distance between electrodes. The test compares
the distance matrix calculated by the two methods: one using the electrode positions by passing the
mne epochs object and the other by passing the electrode positions directly."""

# %% Load Packages
import mne
import numpy as np
from eegrasp import EEGrasp

# %% Load data
subjects = np.arange(1, 10)
runs = [4, 8, 12]

# Download eegbci dataset through MNE

raw_fnames = [mne.datasets.eegbci.load_data(s, runs) for s in subjects]
raw_fnames = np.reshape(raw_fnames, -1)
raws = [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
raw = mne.concatenate_raws(raws)
mne.datasets.eegbci.standardize(raw)
montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage)

# %% Extract electrode positions
eeg_pos = np.array(
    [pos for _, pos in raw.info.get_montage().get_positions()['ch_pos'].items()])

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
evoked = epochs.average()

# %% Initialize EEGrass object and compute distance

# Pass epochs mne object
gsp = EEGrasp(epochs)
Z_epochs = gsp.compute_distance()

gsp = EEGrasp(evoked)
Z_evoked = gsp.compute_distance()

# Pass electrode positions directly
gsp = EEGrasp()
Z = gsp.compute_distance(eeg_pos)

# Assert that the two methods are equivalent
np.testing.assert_array_equal(Z, Z_epochs)
np.testing.assert_array_equal(Z, Z_evoked)

# %% Test Graph Learning usign Epochs

# Initialize EEGrass object
gsp = EEGrasp(epochs)
W, Z = gsp.learn_graph(a=0.1, b=0.1)

# %% Test Graph Learning usign Evoked
# Initialize EEGrass object
gsp = EEGrasp(evoked)
W, Z = gsp.learn_graph(a=0.1, b=0.1)
