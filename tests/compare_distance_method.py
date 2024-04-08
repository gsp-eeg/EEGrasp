# %% Load Packages
import mne
from EEGraSP.EEGraSP import EEGraSP
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from timeit import timeit
#%matplotlib qt
# %%

eegsp = EEGraSP()
montage = mne.channels.make_standard_montage('standard_1005')
ch_names = montage.ch_names
EEG_pos = montage.get_positions()['ch_pos']
# Restructure into array
EEG_pos = np.array([pos for _,pos in EEG_pos.items()])

kdt = spatial.KDTree(EEG_pos)
epsilon = 0.05
#%% Method 2
%%timeit
# Method 1. From pygsp (using scipy)
D, NN = kdt.query(EEG_pos,k=len(EEG_pos),distance_upper_bound=epsilon,
                    p=2,workers=-1)

# Reorder the matrix into the original shape
W = np.zeros(D.shape)
for i,N in enumerate(NN):
    neighbors = D[i,:] != np.inf
    W[i,N[neighbors]] = D[i,neighbors]
np.fill_diagonal(W,np.nan)

#%%
%%timeit
# Method 2. Simpler (in-house method)
W2 = eegsp.euc_dist(EEG_pos)
W2[W2 > epsilon] = 0

# %% Compare results
# Don't compare the diagnonal since np.nan == np.nan is false
# just compare the lowe triangles
tril_indices = np.tril_indices(len(W),-1)

test_result = np.all(W[tril_indices] == W2[tril_indices])

# Plot the resulting matrices

plt.subplot(121)
plt.title('W1: using KDtree.query\nmethod')
plt.imshow(W,vmin=0,vmax=epsilon)
plt.colorbar()

plt.subplot(122)
plt.title('W2: Manual Method')
plt.imshow(W2,vmin=0,vmax=epsilon)
plt.colorbar()

plt.suptitle(f'are W1 and W2 equal?\n{test_result}')

plt.tight_layout()
plt.show()

# %% where is not equal an why?

not_equal_idx = np.where(np.not_equal(W[tril_indices],W2[tril_indices]))[0]

error = (W[tril_indices]-W2[tril_indices])[not_equal_idx]

plt.title('Error')
plt.bar(range(len(not_equal_idx)),error)
#plt.xticks(not_equal_idx)
plt.ylabel('pygsp - inhouse')
plt.show()
# %%
