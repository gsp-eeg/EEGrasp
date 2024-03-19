import numpy as np
from pygsp import graphs

class EEGraSP():

    def __init__(self,data,EEG_pos,ch_names):
        self.data = data
        self.EEG_pos = EEG_pos
        self.ch_names = ch_names
        self.W = None
        self.G = None

    def euc_dist(self,pos):
        """"    
        input: pos -> 2d array of channels by dimensions
        output: 2d array of channels by channels with the euclidean distance. 
        description: compute the euclidean distance between every channel in the array 
        """
        W = np.zeros([pos.shape[0],pos.shape[0]]) # Alocate variable

        for dim in range(pos.shape[1]):
            # Compute the component corresponding to each dimension. Add it to the array
            W += np.square(pos[:,dim][:,None]-pos[:,dim][None,:])
        W = np.sqrt(W)
        
        return W
    
    def compute_weights(self,method='Euclidean'):
        """
        input: method for computing the distance.
        output: weight to be used for the graph computation
        """        
        if method == 'Euclidean':
            W = self.euc_dist(self.EEG_pos)
            np.fill_diagonal(W,np.nan)
        
        self.W = W

    def compute_graph(self,W = None, method='NN',k=5,theta=.2,):
        """"
        input: if W is passed, then the graph is computed. 
        Otherwise the graph will be computed with self.W
        
        output: 

        method: NN -> Nearest Neighbor
                Gaussian -> Gaussian Kernel used based on the self.W matrix
        
        """
        # If passed, used the W matrix
        if W != None:
            W = self.W

        # Check that there is a weight matrix is not a None
        if W == None:
            raise TypeError('Weight matrix cannot be None type')
        try:
            if method=='NN':
                G = graphs.NNGraph(W,NNtype='knn',k=k)
            elif method=='Gaussian':
                G = graphs.Graph(np.exp(-W**2) / (2*theta**2))

            self.G = G

        except:
            print(f'Check arguments needed for the method: {method}')


        
if __name__ == '__main__':
    
    #%% Import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from pygsp import graphs,learning
    from EEGraSP import EEGraSP
    import mne
    from scipy.stats import zscore
    from tqdm import tqdm
    import matplotlib.animation as animation 
    #%matplotlib qt
    
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

    left.plot(gfp=True)
    right.plot(gfp=True)
    plt.show()

    # %% Plot topology maps

    fig,axs = plt.subplots(2,6)
    fig.suptitle('Mental Imaginery',size=16)
    fontdic = {'size':12}
    
    times = np.array([.2,.3,.4,.5,.6])
    window = 0.1
    all_data = np.hstack([left.get_data(),right.get_data()])
    vmin = np.amin(all_data.reshape(-1)) * 1e6 # minimum in time dim
    vmax = np.amax(all_data.reshape(-1)) * 1e6# maximum in time dim
    vlim = (vmin,vmax)
    fig.text(.3,.87,'Left Hand',fontdic)
    fig = left.plot_topomap(times,average=window,axes=axs[0,:],
                            vlim=(vmin,vmax))
    fig.text(.3,.42,'Right Hand',fontdic)
    fig = right.plot_topomap(times,average=window,axes=axs[1,:],
                             vlim=(vmin,vmax))
    
    # Change subplot title
    [ax.set_title(r'{} $\pm {}$ s'.format(t,window),size=9) for ax,t in zip(axs[:,:-1].flatten(),times.repeat(2))]
    
    plt.tight_layout()
    plt.show()

    # %% Animate topomaps
        
    times = np.arange(0,1.1,0.01)
    
    fig,anim = left.animate_topomap(times=times,frame_rate=15,blit=False)
    #anim.save('left.gif',fps=15)
    fig.suptitle('Left Motor Imaginery')
    fig,anim = right.animate_topomap(times=times,frame_rate=15,blit=False)
    #anim.save('right.gif', fps=15)
    
    fig.suptitle('Right Motor Imaginery')


# %% Initialize EEGraph class
    
    eegsp = EEGraSP(right,EEG_pos,ch_names)
    eegsp.compute_weights() # Calculate distance between electrodes
    
    plt.imshow(eegsp.W,cmap='Reds')
    plt.colorbar(label='Euclidean Distance')
    plt.show()

# %% Binarize W based on the histogram's probability mass
    
    def init():
        
        return (dot,im)
    
    def update(frame):
        
        val = np.sort(vec_W)[frame]
        p = prob[frame]

        th_W = W > val # Keep distances lower than the threshold
        np.fill_diagonal(th_W,0) # No self loops

        dot.set_offsets([val,p])  
        im.set_data(th_W) 

        axs[1,1].clear()
        G = graphs.Graph(th_W)
        G.set_coordinates()
        G.plot(ax=axs[1,1])
        

        return (dot,im)


    W = eegsp.W.copy()
    tril_idx = np.tril_indices(len(W),-1)
    vec_W = W[tril_idx] 
    count,bins = np.histogram(vec_W,bins=len(vec_W),density=False)
    
    prob = np.cumsum(count/len(vec_W))

    # Initiate figure
    fig,axs = plt.subplots(2,2,figsize=(7,7))
    axs[0,0].set_title('Cumulative Distribution Function')
    axs[0,0].set_xlabel('Euc. Distance')
    axs[0,0].set_ylabel('Probability')

    lines, = axs[0,0].plot(np.sort(vec_W),prob)
    dot = axs[0,0].scatter(np.sort(vec_W)[0],prob[0])
    hist = axs[1,0].hist(vec_W,bins=len(vec_W),density=True)
    th_W = W > 0
    im = axs[0,1].imshow(th_W,cmap='gray')
    fig.colorbar(im,ax=axs[0,1])

    anim = animation.FuncAnimation(fig,update,
                                   frames=range(len(prob))[::20],
                                   interval=1,blit=False,
                                   cache_frame_data=True)

# %% Compute graph based on nearest neighbors based on euc. distance
    
    data = epochs['left'].get_data()

    vk = np.arange(3,10) # Number of nearest neighbors to try

    nchannels = data.shape[1]
    nsamples = data.shape[2]
    nepochs = data.shape[0] 
    
    missing_idx = 5
    
    measures = data.copy()
    mask = np.ones(len(EEG_pos)).astype(bool)
    mask[missing_idx] = False
    measures[:,~mask,:] = np.nan
    
    # %% Interpolate signal

    # Allocate error matrix of len(vk) x epochs x timepoints
    error = np.zeros([len(vk),measures.shape[0]])
    recovery = np.zeros([len(vk),nepochs,nsamples])
    for i,k in enumerate(tqdm(vk)):
        # Compute graph from EEG distance
        eegsp.compute_graph(k=k)

        # Reconstruct every epoch
        for ii,epoch in enumerate(measures):

            # Reconstruct every timepoint
            for iii,t in enumerate(epoch.T):

                # Recover a signal                
                recovery[i,ii,iii] = learning.regression_tikhonov(eegsp.G, t, mask, tau=0)[missing_idx]
            
            error[i,ii] = (np.linalg.norm(data[ii,missing_idx,:] - recovery[i,ii,:]))

    # %% Plot
    ii = 1
    plt.plot(recovery[4,ii,:])
    plt.plot(data[ii,missing_idx,:])
    plt.show()

# %% Plot error
    best_k = vk[np.argmin(error,axis=0)]
    plt.plot(vk,error)
    plt.show()





