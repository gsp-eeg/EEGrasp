""""
This code tests the graph construction by comparing to the results
obtained with PyGSP2.
"""

# %% Load libraries
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from pygsp2 import graphs, learning
from eegrasp import EEGrasp

# %% Load Data
df_pos = pd.read_pickle('df_pos.pkl')
# Remove EOC channel. Shouldn't be here. Investigate: I think we're including
# the EOC channels
df_pos = df_pos.drop(index=59)

df_100 = pd.read_pickle('df_100.pkl')
signal = df_100.iloc[0, 1:].to_numpy()

X = df_pos[['x', 'y']].to_numpy()

# Make a mesurement with missing channels
mask = np.ones(len(signal), dtype=bool)
MISSING_IDX = 5
mask[MISSING_IDX] = False
measures = signal.copy()
measures[~mask] = np.nan

# %% Standard Method
rbfi = interp.Rbf(X[mask, 0], X[mask, 1], measures[mask])
error_rbf = np.abs(rbfi(X[~mask, 0], X[~mask, 1]) - signal[~mask])[0]

# %% PyGSP2 NNGraph
sigmas = np.linspace(0.02, 0.08, 200)
errors = np.zeros(len(sigmas))
for i, sigma in enumerate(sigmas):
    G = graphs.NNGraph(X, 'radius', sigma=sigma,
                       epsilon=0.5, rescale=True, center=True)
    G.estimate_lmax()
    # Solve the classification problem by reconstructing the signal:
    recovery = learning.regression_tikhonov(G, measures, mask, tau=0)
    error = np.linalg.norm(signal[~mask] - recovery[~mask])
    errors[i] = error

# %% Inhouse Method

eeggsp = EEGrasp(measures[:, None], X)
distance = eeggsp.compute_distance()

sigmas2 = np.sqrt(sigmas/2)
errors2 = np.zeros(len(sigmas2))
for i, sigma in enumerate(sigmas2):
    eeggsp.compute_graph(epsilon=0.5, sigma=sigma)
    recovery = eeggsp.interpolate_channel(missing_idx=MISSING_IDX)
    error = np.linalg.norm(signal[~mask] - recovery[~mask])
    errors2[i] = error

# %% Plot error
plt.close('all')
plt.plot(sigmas, errors, color='blue', label='PyGSP2')
plt.plot(sigmas, errors2, color='magenta', label='EEGRASP')
plt.xlabel(r'$\sigma$')
plt.xticks(sigmas[::22],
           labels=[f'{s:0.2f}\n{s2:0.2f}' for s, s2 in zip(sigmas[::22],
                                                           sigmas2[::22])])
plt.axhline(error_rbf, color='red', label='RBF interpolation', linestyle='--')
plt.axhline(0.02, color='purple', label='y = 0.0003', linestyle='--')
plt.grid()
plt.legend()
plt.show()

# %% Error
print(f'PyGSP2: {np.amin(errors):.1e}')
print(f'EEGRASP: {np.amin(errors2):.1e}')
print(f'Error percentage: {(np.amin(errors2)/np.amin(errors) * 100):.1f}%')

print('\nNow call NNGraph with rescale=False and center=False')
# Normalize distances
X_norm = X - np.amin(X)
X_norm = X_norm / np.amax(X_norm)

sigmas = np.linspace(0.02, 0.08, 200)
errors = np.zeros(len(sigmas))
for i, sigma in enumerate(sigmas):
    G = graphs.NNGraph(X_norm, 'radius', sigma=sigma,
                       epsilon=0.5, rescale=False, center=False)
    G.estimate_lmax()
    # Solve the classification problem by reconstructing the signal:
    recovery = learning.regression_tikhonov(G, measures, mask, tau=0)
    error = np.linalg.norm(signal[~mask] - recovery[~mask])
    errors[i] = error

# %% Plot error
plt.figure()
plt.plot(sigmas, errors, color='blue', label='PyGSP2 - No rescale, No Center')
plt.plot(sigmas, errors2, color='magenta', label='EEGRASP')
plt.xlabel(r'$\sigma$')
plt.xticks(sigmas[::22],
           labels=[f'{s:0.2f}\n{s2:0.2f}' for s, s2 in zip(sigmas[::22],
                                                           sigmas2[::22])])
plt.axhline(error_rbf, color='red', label='RBF interpolation', linestyle='--')
plt.axhline(0.02, color='purple', label='y = 0.02', linestyle='--')
plt.grid()
plt.legend()
plt.show()

# %% Error
print(f'PyGSP2: {np.amin(errors):.1e}')
print(f'EEGRASP: {np.amin(errors2):.1e}')
print(f'Error percentage: {(np.amin(errors2)/np.amin(errors) * 100):.1f}%')
