#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs,learning
from EEGraSP.EEGraSP import EEGraSP
import mne
from scipy.stats import zscore
from tqdm import tqdm
import matplotlib.animation as animation 
%matplotlib qt

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

fig_left = left.plot(gfp=True)
fig_right = right.plot(gfp=True)

#fig_left.savefig('left_erp.png')
#fig_right.savefig('rihgt_erp.png')

plt.show()

# %% Plot topology maps

# Times to plot
times = np.arange(0.2,1.2,.2).round(1)
window = 0.1 # Window length for averaging
# Look vor vmin and vmax to share scale
all_data = np.hstack([left.get_data(),right.get_data()])
vmin = np.amin(all_data.reshape(-1)) * 1e6
vmax = np.amax(all_data.reshape(-1)) * 1e6
vlim = (vmin,vmax)

fig,axs = plt.subplots(2,len(times)+1,figsize=(8,4))
fig.suptitle('Mental Imaginery',size=16)
fontdic = {'size':12}


# Make plots and titles
fig.text(.3,.87,'Left Hand',fontdic)
fig = left.plot_topomap(times,average=window,axes=axs[0,:],
                        vlim=(vmin,vmax))
fig.text(.3,.42,'Right Hand',fontdic)
fig = right.plot_topomap(times,average=window,axes=axs[1,:],
                            vlim=(vmin,vmax))

# Change subplots titles
for ax,t in zip(axs[:,:-1].flatten('F'),times.repeat(2)):
    ax.set_title(r'{} $\pm {}$ s'.format(t,window),size=9)

fig.tight_layout()
#fig.savefig('LR_topoplots.png')
fig.show()

# %% Animate topomaps
    
times = np.arange(0,1.1,0.01)

# Animation for left-hand imaginery
fig,anim = left.animate_topomap(times=times,frame_rate=15,blit=False)
fig.suptitle('Left Motor Imaginery')
# Uncomment to save
#anim.save('left.gif',fps=15)

# Animation for right-hand imaginery
fig,anim = right.animate_topomap(times=times,frame_rate=15,blit=False)
fig.suptitle('Right Motor Imaginery')
# Uncomment to save
#anim.save('right.gif', fps=15)


# %% Initialize EEGraph class

eegsp = EEGraSP(right,EEG_pos,ch_names)
W = eegsp.compute_distance() # Calculate distance between electrodes

# Plot euclidean distance between electrodes
fig,ax = plt.subplots()
ax.set_title('Electrode Distance')
im = ax.imshow(W,cmap='Reds')
fig.colorbar(im,label='Euclidean Distance')

# Uncomment to save
#fig.savefig('euc_dist.png')
fig.show()

# %% Binarize W based on the histogram's probability mass

W = eegsp.W.copy()
tril_idx = np.tril_indices(len(W),-1)
vec_W = W[tril_idx] 
count,bins = np.histogram(vec_W,bins=len(vec_W),density=False)

prob = np.cumsum(count/len(vec_W))
th_W = W > 0

# Initiate figure
fig,axs = plt.subplots(2,2,figsize=(10,9))
axs[0,0].set_title('Cumulative Distribution Function')
axs[0,0].set_xlabel('Euc. Distance')
axs[0,0].set_ylabel('Probability')

lines, = axs[0,0].plot(np.sort(vec_W),prob,color='black')
dot = axs[0,0].scatter(np.sort(vec_W)[0],prob[0],c='red')

hist = axs[1,0].hist(vec_W,bins=20,density=True,color='teal')
vline = axs[1,0].axvline(np.amin(vec_W),color='purple')
axs[1,0].set_title('Histogram')
axs[1,0].set_xlabel('Euc. Distance')
axs[1,0].set_ylabel('Probability Density')

im = axs[0,1].imshow(th_W,cmap='gray')
axs[0,1].set_xlabel('Electrode')
axs[0,1].set_ylabel('Electrode')
axs[0,1].set_title('Adjacency Matrix')
cbar = fig.colorbar(im,ax=axs[0,1],ticks = [0,1])
cbar.ax.set_yticklabels(['Unconnected','Connected'])

fig.tight_layout()

# Define function for animation
def update(frame):
    
    val = np.sort(vec_W)[frame]
    p = prob[frame]

    th_W = W <= val # Keep distances lower than the threshold
    np.fill_diagonal(th_W,0) # No self loops

    dot.set_offsets([val,p])  
    im.set_data(th_W) 
    vline.set_data([[val,val],[0,1]])

    axs[1,1].clear()
    G = graphs.Graph(th_W)
    G.set_coordinates()
    G.plot(ax=axs[1,1])

    return (dot,im)

anim = animation.FuncAnimation(fig,update,
                                frames=np.arange(len(prob))[::8],
                                interval=1,blit=False,
                                cache_frame_data=False)

# Uncomment to save animation 
#anim.save('G_thr.gif',fps=30)


# %% Compute graph based on nearest neighbors based on euc. distance

data = epochs['left'].get_data()

nchannels = data.shape[1]
nsamples = data.shape[2]
nepochs = data.shape[0] 

missing_idx = 5

measures = data.copy()
mask = np.ones(len(EEG_pos)).astype(bool)
mask[missing_idx] = False
measures[:,~mask,:] = np.nan

# %% Graph based on gaussian kernel

epsilon = np.sort(np.unique(vec_W))
for e in epsilon:

    G = graphs.NNGraph(EEG_pos,'radius',rescale=False,epsilon=e)
    W = G.W.toarray()

    fig,axs = plt.subplots(1,2,figsize=(7,4))
    
    im = axs[0].imshow(W,'gray',vmin=0,vmax=1)
    fig.colorbar(im,cmap='jet')

    G.set_coordinates()
    G.plot(ax=axs[1])

# %% Graph with distances thresholded
from scipy import spatial

kdt = spatial.KDTree(EEG_pos)
epsilon = 0.05

# Method 1. From pygsp (using scipy)
D, NN = kdt.query(EEG_pos,k=len(EEG_pos),distance_upper_bound=epsilon,
                    p=2)

# Reorder the matrix into the original shape
W = np.zeros(D.shape)
for i,N in enumerate(NN):
    neighbors = D[i,:] != np.inf
    W[i,N[neighbors]] = D[i,neighbors]
np.fill_diagonal(W,np.nan)

# Method 2. Simpler (in-house method)

W2 = eegsp.compute_distance(EEG_pos,method='Euclidean')

W2[W2 > epsilon] = 0

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



