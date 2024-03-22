""""
creation date: 21/03/2024
Author: jrodino14@gmail.com
This script defines the main class in the EEGRraSP package.
No inputs are required for the class to initialize, though, for the
computation of graphs and fitting to functional data: EEG_pos,
data and ch_names are required.
"""
import numpy as np
from pygsp import graphs,learning
from tqdm import tqdm

class EEGraSP():

    def __init__(self,data=None,EEG_pos=None,ch_names=None):
        
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
        W = np.zeros([pos.shape[0],pos.shape[0]],dtype=np.float64) # Alocate variable
        pos = pos.astype(float)
        for dim in range(pos.shape[1]):
            # Compute the component corresponding to each dimension. Add it to the array
            W += np.power(pos[:,dim][:,None]-pos[:,dim][None,:],2)
        W = np.sqrt(W)
        
        return W
    
    def gaussian_kernel(self,X,sigma=0.1):
        return np.exp(-np.power(X, 2) / (2.*np.power(float(sigma),2)))
    
    def compute_distance(self,pos=None,method='Euclidean'):
        """
        descrpition: method for computing the distance.
        output: weight to be used for the graph computation
        """

        # If passed, used the W matrix
        if type(pos) == type(None):
            pos = self.EEG_pos

        if method == 'Euclidean':
            W = self.euc_dist(pos)
            np.fill_diagonal(W,np.nan)
        
        self.W = W

        return W


    def compute_graph(self,W = None, method='Gaussian',k=5,epsilon=.1,sigma=.1):
        """"
        W -> if W is passed, then the graph is computed. 
        Otherwise the graph will be computed with self.W.
        W should correspond to a non-sparse 2-D array.
        Epsilon -> maximum distance to threshold the array.
        sigma -> Sigma parameter for the gaussian kernel.
        
        method: NN -> Nearest Neighbor
                Gaussian -> Gaussian Kernel used based on the self.W matrix

        output: Graph structure from pygsp

        
        """
        # If passed, used the W matrix
        if type(W) == type(None):
            W = self.W

        # Threshold matrix
        W[W>=epsilon] = 0

        # Check that there is a weight matrix is not a None
        if type(W) == type(None):
            raise TypeError('Weight matrix cannot be None type')
        try:
            if method=='NN':
                G = graphs.NNGraph(W,NNtype='knn',k=k)

            elif method=='Gaussian':
                weights = self.gaussian_kernel(W,sigma=sigma)
                weights[W == 0] = 0
                np.fill_diagonal(weights,0)
                G = graphs.Graph(weights)

            self.G = G
            return G

        except:
            raise TypeError(f'Check arguments needed for the method: {method}')

    def fit_graph_to_data(self,data=None,W=None,sigma=0.1,
                          missing_idx=None,
                          weight_method='Gaussian',
                          error_method='MRSE'):
        """"
        Description: This method returns the graph that best reconstructed the entire segment of data.
        It will itterate through all the unique values of the distance matrix.
        data -> 2-dimensional array. The first dim. is Channels 
        and second time. It can be passed to the instance class or the method

        W -> Unthresholded distance matrix (2-dimensional array). It can be passed to the instance of
        the class or method.
        sigma -> parameter of the Gaussian Kernel transformation
        error_method -> string with the error to be used to compare the interpolation results.
        """
        # Check if values are passed or use the instance's
        if type(W) == type(None):
            W = self.W
        if type(data) == type(None):
            data = self.data
        
        if (type(W) == type(None)) or (type(data) == type(None)):
            raise TypeError('Check data or W arguments.')
        elif (type(missing_idx)==type(None)):
            raise TypeError('Parameter missing_idx not specified.')

        # Vectorize the distance matrix
        tril_indices = np.tril_indices(len(W),-1)
        vec_W = W[tril_indices]

        # Sort and extract unique values
        distances = np.sort(np.unique(vec_W))

        # Create time array
        time = np.arange(0,data.shape[-1])

        # Mask to ignore missing channel
        mask = np.ones(data.shape[-2]).astype(bool)
        mask[missing_idx] = False
        # Simulate eliminating the missing channel
        signal = data.copy()
        signal[:,missing_idx] = np.nan

        # Allocate array to reconstruct the signal
        recovery = np.zeros([len(distances),len(time)])

        # Allocate Error array
        error = np.zeros([len(distances)])
                
        # Loop to look for the best parameter
        for i,epsilon in enumerate(tqdm(distances)):
            # Compute thresholded weight matrix
            G = self.compute_graph(W,method=weight_method,
                               epsilon=epsilon,sigma=0.1)
            # Interpolate signal, iterating over time
            for t in enumerate(time):
                recovery[i,t] = learning.regression_tikhonov(G, signal[:,t], mask, tau=0)[missing_idx]
            
            error[i] = np.linalg.norm(data[missing_idx,:]-recovery[i,:])
        
        best_epsilon = distances[np.argmin(error)] 
        
        results = {'Error':error,'Signal':recovery,'best_epsilon':best_epsilon}
        
        return results
        
    #def interpolate_channel(self,data,inter_method=''):

        
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

    fig_left = left.plot(gfp=True)
    fig_right = right.plot(gfp=True)
    
    fig_left.savefig('left_erp.png')
    fig_right.savefig('rihgt_erp.png')

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
    fig.savefig('LR_topoplots.png')
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
    eegsp.compute_distance() # Calculate distance between electrodes
    W = eegsp.W.copy()

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

    epsilon = np.arange(0.04,0.2,0.01)
    for e in epsilon:

        G = graphs.NNGraph(EEG_pos,'radius',rescale=False,epsilon=e)
        W = G.W.toarray()

        fig,axs = plt.subplots(1,2,figsize=(7,4))
        
        im = axs[0].imshow(W,'gray',vmin=0,vmax=1)
        fig.colorbar(im,cmap='jet')

        G.set_coordinates()
        G.plot(ax=axs[1])

    # %% Interpolate signal

    vk = np.arange(3,10)
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





