import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from random import randrange
import math
import utilities as u
import preprocessing as pp
import matplotlib.gridspec as gridspec



def plot_top_cells(S_tm,masks,SI,morph,maxcells=400):
    '''
    plot single cell trial x position rate maps for the cells with highest spatial information
    inputs: S_tm - trials x positions x neurons tensor of activity
            masks - dictionary of masks to find place cells (output of place_cells_calc)
            SI - spatial information of each cell (output of place_cells_calc)
            morph - vector of morph values for each trial
            maxcells - maximum number of cells to try to plot
    outputs: f - figure object for saving
    '''

    # find cells that are place cells in any of the morphs
    allmask = masks[0]
    for k,v in masks.items():
        allmask = allmask | v

    # number of cells to plot
    nplacecells = np.minimum(allmask.sum(),maxcells)

    # set up axes
    xstride = 3
    ystride = 4
    nperrow = 8
    f = plt.figure(figsize=[nperrow*xstride,nplacecells/nperrow*ystride])
    gs = gridspec.GridSpec(math.ceil(nplacecells/nperrow)*ystride,xstride*nperrow)

    # sum spatial information across morphs and use as order for cells to be plotted
    SI_total = [SI[m]['all'] for m in SI.keys()]
    SIt = SI_total[0]
    for ind in SI_total[1:]:
        SIt+=ind
    si_order = np.argsort(SIt)[::-1]

    morph_order = np.argsort(morph)
    morph_s = morph[morph_order]

    for cell in range(nplacecells): # for each cell
        # do some smoothing in position axis for visualization
        c = u.nansmooth(np.squeeze(S_tm[:,:,si_order[cell]]),[0,3])
        # normalize by mean
        c/=np.nanmean(c.ravel())
        # add plots
        row_i = int(ystride*math.floor(cell/nperrow))
        col_i = int(xstride*(cell%nperrow))



        # plot in morph order
        morphsort_ax = f.add_subplot(gs[row_i:row_i+ystride-1,col_i])
        morphsort_ax.imshow(c[morph_order,:],cmap='magma',aspect='auto')
        tick_labels = ["%.2f" % morph_s[i] for i in tick_inds]

        # plot in trial order
        trialsort_ax = f.add_subplot(gs[row_i:row_i+ystride-1,col_i+1])
        trialsort_ax.imshow(c,cmap='magma',aspect='auto')
        tick_inds = np.arange(0,c.shape[0],10)


        morphsort_ax.set_yticks(tick_inds)
        morphsort_ax.set_yticklabels(tick_labels,fontsize=5)
        trialsort_ax.set_yticks([])
        morphsort_ax.set_xticks([])
        trialsort_ax.set_xticks([])
        morphsort_ax.set_title("%d" % si_order[cell])

        if row_i==0 and col_i==0:
            trialsort_ax.set_ylabel('Trial #')
            trialsort_ax.yaxis.set_label_position('right')
            morphsort_ax.set_ylabel('Mean Morph')

    return f

def reward_cell_scatterplot(fr0, fr1, rzone0 = [250,315], rzone1 = [350,415],tmax= 450):
    '''
    plot peak location  of place field in one environment vs the other. reward zones in
    context discrimination task are highlighted
    inputs: fr0 - [pos,neurons] array of average firing rate for morph 0 trials
            fr1 - [pos, neurons] array of average firing rat for morph 1 trials
            rzone0 - bounds of morph 0 reward zone
            rzone1 - bound of morph 1 reward zone
            tmax  - max position on track
    outpus: f - figure object
            ax - axis array
    '''

    f = plt.figure(figsize=[10,10])
    gs = gridspec.GridSpec(5,5)
    ax = f.add_subplot(gs[0:-1,0:-1])


    # plot argmax of pos in morph 0 vs argmax of pos in morp 1
    ax.scatter(5.*np.argmax(fr0,axis=0),5*np.argmax(fr1,axis=0),color='black')
    ax.plot(np.arange(tmax),np.arange(tmax),color='black')

    # add reward zones
    ax.fill_between(np.arange(tmax),rzone0[0],y2=rzone0[1],color=plt.cm.cool(0),alpha=.2)
    ax.fill_betweenx(np.arange(tmax),rzone0[0],x2=rzone0[1],color=plt.cm.cool(0),alpha=.2)
    ax.fill_between(np.arange(tmax),rzone1[0],y2=rzone1[1],color=plt.cm.cool(1.),alpha=.2)
    ax.fill_betweenx(np.arange(tmax),rzone1[0],x2=rzone1[1],color=plt.cm.cool(1.),alpha=.2)

    # add histogram of argmax's
    ax1 = f.add_subplot(gs[-1,0:-1])
    ax1.hist(5.*np.argmax(fr0,axis=0),np.arange(0,tmax+10,10))
    ax1.fill_betweenx(np.arange(40),rzone0[0],x2=rzone0[1],color=plt.cm.cool(0),alpha=.2)
    ax1.fill_betweenx(np.arange(40),rzone1[0],x2=rzone1[1],color=plt.cm.cool(1.),alpha=.2)

    # add histogram of argmax's
    ax2 = f.add_subplot(gs[0:-1,-1])
    ax2.hist(5.*np.argmax(fr1,axis=0),np.arange(0,tmax+10,10),orientation='horizontal')
    ax2.fill_between(np.arange(40),rzone0[0],y2=rzone0[1],color=plt.cm.cool(0),alpha=.2)
    ax2.fill_between(np.arange(40),rzone1[0],y2=rzone1[1],color=plt.cm.cool(1.),alpha=.2)

    return f, ax


def common_cell_remap_heatmap(fr0, fr1, rzone = [225,400], tmax= 450,bin_size=10.,norm = True):
    '''
    plot histogram of peak locations  of place field in one environment vs the other. reward zone is highlighted
    inputs: fr0 - [pos,neurons] array of average firing rate for morph 0 trials
            fr1 - [pos, neurons] array of average firing rat for morph 1 trials
            rzone - bounds of possible reward zone
            tmax  - max position on track
            bin_size - width of spatial bins
            norm - normalize occupancy
    outpus: f - figure object
            ax - axis array
    '''


    f = plt.figure(figsize=[10,10])
    gs = gridspec.GridSpec(5,5)
    ax = f.add_subplot(gs[0:-1,0:-1])

    # for every cell, find location of peak firing in different environments
    heatmap = np.zeros([fr0.shape[0],fr1.shape[0]])
    for cell in range(fr0.shape[1]):
        heatmap[np.argmax(fr0[:,cell],axis=0),np.argmax(fr1[:,cell],axis=0)]+=1

    # normalize for comparison across mice/conditions
    if norm:
        ax.imshow(heatmap.T/np.amax(heatmap),cmap='magma',vmin=0.05,vmax=.15)#,aspect='auto')
    else:
        ax.imshow(heatmap.T,cmap='magma',vmax=.7*np.amax(heatmap.ravel()))

    ## add marginal histograms
    ax1 = f.add_subplot(gs[-1,0:-1],sharex=ax)
    hist,edges = np.histogram(bin_size*np.argmax(fr0,axis=0),np.arange(0,tmax+10,10))
    ax1.fill_between(np.linspace(0,45,num=edges.shape[0]-1)-.5,hist/hist.sum(),color=plt.cm.cool(1.))
    ax1.set_xlim([-1,46])
    ax1.fill_betweenx([0,.05],rzone[0]/bin_size,x2=rzone[1]/bin_size,color='black',alpha=.2)


    ax2 = f.add_subplot(gs[0:-1,-1],sharey=ax)
    hist,edges = np.histogram(bin_size*np.argmax(fr1,axis=0),np.arange(0,tmax+10,10))
    ax2.fill_betweenx(np.linspace(0,45,num=edges.shape[0]-1)-.5,hist/hist.sum(),color=plt.cm.cool(0.))
    ax2.set_ylim([-1,46])
    ax2.fill_between([0,.05],rzone[0]/bin_size,y2=rzone[1]/bin_size,color='black',alpha=.2)


    return f, ax


def spatial_info(frmap,occupancy):
    '''calculate spatial information bits/spike
    inputs: frmap - [spatial bins, cells] spatially binned activity rate for each cell
            occupancy - [spatial bins,] fractional occupancy of each spatial bin
    outputs: SI - [cells,] spatial information for each cell '''

    ### vectorizing
    P_map = frmap - np.amin(frmap)+.001 # make sure there's no negative activity rates
    P_map = P_map/P_map.mean(axis=0)
    SI = ((P_map*occupancy[:,np.newaxis])*np.log2(P_map)).sum(axis=0) # Skaggs and McNaughton spatial information

    return SI



def place_cells_calc(C, position, trial_info, tstart_inds,
                teleport_inds,pthr = .99,speed=None,win_trial_perm=True,morphlist = [0,1]):
    '''Find cells that have significant spatial information info. Use bootstrapped estimate of each cell's
    spatial information to minimize effect of outlier trials
    inputs:C - [timepoints, neurons] activity rate/dFF over whole session
            position - [timepoints,] animal's position on the track at each timepoint
            trial_info - trial information dict from u.by_trial_info()
            tstart_inds - [ntrials,] array of trial start times (can filter by which trials you want to
                use for calculation)
            teleport_inds - [ntrials,] array of trial stop times
            pthr - (float) 1 - p-value for shuffle procedure
            speed - [timepoints,] or None; animal's speed at each timepoint. Used for filtering stationary
                timepoints. If None, no filtering is performed
            win_trial_perm - (bool); whether to perform shuffling within a trial. If false,
                shuffling is performed with respect to the entire timeseries
            morphlist - (list); which mean morph values to use in separate place cell calculations
    outputs: masks - dictionary of masks for cells with significant spatial info in each morph
            FR - dictionary of firing rate maps per cell per morph
            SI - dictionary of spatial information per cell per morph

    '''

    # get by trial info
    C_trial_mat, occ_trial_mat, edges,centers = u.make_pos_bin_trial_matrices(C,position,tstart_inds,teleport_inds,speed = speed)
    morphs = trial_info['morphs'] # mean morph values

    # split into morph specific arrays
    C_morph_dict = u.trial_type_dict(C_trial_mat,morphs)
    occ_morph_dict = u.trial_type_dict(occ_trial_mat,morphs)
    tstart_inds, teleport_inds = np.where(tstart_inds==1)[0], np.where(teleport_inds==1)[0]
    tstart_morph_dict = u.trial_type_dict(tstart_inds,morphs)
    teleport_morph_dict = u.trial_type_dict(teleport_inds,morphs)

    # for each morph value
    FR,masks,SI = {}, {}, {}
    for m in morphlist:

        FR[m]= {}
        SI[m] = {}

        # firing rate maps
        FR[m]['all'] = np.nanmean(C_morph_dict[m],axis=0)
        occ_all = occ_morph_dict[m].sum(axis=0) # fractional occupancy
        occ_all /= occ_all.sum()
        SI[m]['all'] =  spatial_info(FR[m]['all'],occ_all) # spatial information
        n_boots=30

        tmat = C_morph_dict[m] # trial x position x cell activity rates
        omat = occ_morph_dict[m] # single trial occupancy
        SI_bs = np.zeros([n_boots,C.shape[1]]) # bootstrapped spatial information
        print("start bootstrap")
        for b in range(n_boots):

            # pick a random subset of trials
            ntrials = tmat.shape[0]
            bs_pcnt = .67 # proportion of trials to keep
            bs_thr = int(bs_pcnt*ntrials) # number of trials to keep

            # calculate spatial information with randomly selected set of trials
            bs_inds = np.random.permutation(ntrials)[:bs_thr]
            FR_bs = np.nanmean(tmat[bs_inds,:,:],axis=0)
            occ_bs = omat[bs_inds,:].sum(axis=0)
            occ_bs/=occ_bs.sum()
            SI_bs[b,:] = spatial_info(FR_bs,occ_bs)
        print("end bootstrap")
        SI[m]['bootstrap']= np.median(SI_bs,axis=0).ravel() # take the true SI as the median of the bootstrapped estimates
        p_bs, shuffled_SI = spatial_info_perm_test(SI[m]['bootstrap'],C, # permutation test
                                position,tstart_morph_dict[m],teleport_morph_dict[m],
                                nperms=1000,win_trial=win_trial_perm)
        masks[m] = p_bs>pthr # hypothesis test

    return masks, FR, SI



def spatial_info_perm_test(SI,C,position,tstart,tstop,nperms = 10000,shuffled_SI=None,win_trial = True):
    '''run permutation test on spatial information calculations and return empirical p-values for each cell
    inputs: SI - [ncells,] array of 'true' spatial information for each cell
            C - [ntimepoints,cells] activity rate of each cell over whole session
            position - [ntimepoints,] array of animal positions at each timepoint
            tstart - [ntrials,] array of trial start indices to be considered in shuffle
            tstop - [ntrials,] array of trial stop indices to be considered in shuffle
            perms - (float/int); number of permutations to run
            shuffled_SI - [ncells,permutations] or None; if shuffled distribution already exists, just
                calculate p - values
            win_trial - bool; whether to do shuffling within each trial or across whole timeseries
    returns p - [ncells,] array of 1 - p-values from perm test
            shuffled_SI - [ncells, nperms] shuffled distribution
    '''
    if len(C.shape)<2: # if only considering one cell, expand dimensions
        C = C[:,np.newaxis]

    if shuffled_SI is None:
        shuffled_SI = np.zeros([nperms,C.shape[1]])
        for perm in range(nperms): # for each permutation

            if win_trial: # within trial permuation
                C_tmat, occ_tmat, edes,centers = u.make_pos_bin_trial_matrices(C,position,tstart,tstop,perm=True)
            else:
                C_perm = np.roll(C,randrange(30,position.shape[0],30),axis=0) # perform permutation over whole time series
                C_tmat, occ_tmat, edes,centers = u.make_pos_bin_trial_matrices(C,position,tstart,tstop,perm=False)

            fr, occ = np.squeeze(np.nanmean(C_tmat,axis=0)), occ_tmat.sum(axis=0) # average firing rate and occupancy
            occ/=occ.sum()

            si = spatial_info(fr,occ) # shuffled spatial information
            shuffled_SI[perm,:] = si


    # calculate p-values
    p = np.zeros([C.shape[1],])
    for cell in range(C.shape[1]):
        p[cell] = np.sum(SI[cell]>shuffled_SI[:,cell])/nperms

    return p, shuffled_SI

def plot_placecells(C_morph_dict,masks,cv_sort=True, plot = True):
    '''plot place place cells across morph values using a cross-validated population sorting
    inputs: C_morph_dict - output from u.trial_type_dict(C_trial_mat, morphs) where C is the [trials, positions,ncells]
                and morphs is [ntrials,] array of mean morph values
            masks - dictionary of place cell masks from place_cells_calc
            cv_sort - bool; calculate sorting from all trials (False) or a randomly selected half of trials (True)
            plot - bool; whether or not to actually generate matplotlib plots or just return sorted data
    outputs: f - figure handle
            ax - axis array
            PC_dict - dictionary of cross-val sorted population activity rate maps'''

    morphs = [k for k in C_morph_dict.keys() if isinstance(k,np.float64)]
    if plot:
        f,ax = plt.subplots(2,len(morphs),figsize=[5*len(morphs),15])
        f.subplots_adjust(wspace=.01,hspace=.05)

    getSort = lambda fr : np.argsort(np.argmax(np.squeeze(np.nanmean(fr,axis=0)),axis=0))
    PC_dict = {}
    PC_dict[0],PC_dict[1] = {},{}
    if cv_sort:

        # get sort for 0 morph from random half of trials
        ntrials0 = C_morph_dict[0].shape[0]
        sort_trials_0 = np.random.permutation(ntrials0)
        ht0 = int(ntrials0/2)
        arr0 = C_morph_dict[0][:,:,masks[0]]
        arr0 = arr0[sort_trials_0[:ht0],:,:]
        sort0 = getSort(arr0)

        _arr0 = np.copy(arr0)
        _arr0[np.isnan(arr0)]=0.
        norm0 = np.amax(np.nanmean(_arr0,axis=0),axis=0) # normalization from training data

        # do the same for 1 morph
        ntrials1 = C_morph_dict[1].shape[0]
        ht1 = int(ntrials1/2)
        sort_trials_1 = np.random.permutation(ntrials1)
        arr1= C_morph_dict[1][:,:,masks[1]]
        arr1 = arr1[sort_trials_1[:ht1],:,:]
        sort1 = getSort(arr1)

        _arr1 = np.copy(arr1)
        _arr1[np.isnan(arr1)]=0.
        norm1= np.amax(np.nanmean(_arr1,axis=0),axis=0)


    else: # otherwise, use all trials
        sort0 = getSort(C_morph_dict[0][:,:,masks[0]])
        norm0 = np.amax(np.nanmean(C_morph_dict[0][:,:,masks[0]],axis=0),axis=0)
        sort1 = getSort(C_morph_dict[1][:,:,masks[1]])
        norm1 = np.amax(np.nanmean(C_morph_dict[1][:,:,masks[1]],axis=0),axis=0)


    for i,m in enumerate(morphs):
        if cv_sort: # get rate maps for other half of trials
            if m ==0:
                fr_n0 = np.squeeze(np.nanmean(C_morph_dict[m][sort_trials_0[ht0:],:,:],axis=0))
                fr_n1 = np.squeeze(np.nanmean(C_morph_dict[m],axis=0))
            elif m==1:
                fr_n1 = np.squeeze(np.nanmean(C_morph_dict[m][sort_trials_1[ht1:],:,:],axis=0))
                fr_n0 = np.squeeze(np.nanmean(C_morph_dict[m],axis=0))
            else:
                fr_n0 = np.squeeze(np.nanmean(C_morph_dict[m],axis=0))
                fr_n1 = np.squeeze(np.nanmean(C_morph_dict[m],axis=0))
        else: # get rate maps
            fr_n0 = np.squeeze(np.nanmean(C_morph_dict[m],axis=0))
            fr_n1 = np.squeeze(np.nanmean(C_morph_dict[m],axis=0))

        # smooth and normalize by within cell max
        fr_n0, fr_n1 = fr_n0[:,masks[0]], fr_n1[:,masks[1]]
        fr_n0= gaussian_filter1d(fr_n0/norm0,2,axis=0)
        fr_n1 = gaussian_filter1d(fr_n1/norm1,2,axis=0)

        # save data
        fr_n0, fr_n1 = fr_n0[:,sort0], fr_n1[:,sort1]
        PC_dict[0][m], PC_dict[1][m]= fr_n0.T, fr_n1.T
        if plot:
            ax[0,i].imshow(fr_n0.T,aspect='auto',cmap='pink',vmin=0.2,vmax=.9)
            ax[1,i].imshow(fr_n1.T,aspect='auto',cmap='pink',vmin=0.2,vmax=.9)
            if i>0:
                ax[0,i].set_yticks([])
                ax[1,i].set_yticks([])
            ax[0,i].set_xticks([])
            ax[1,i].set_xticks([])

    if plot:
        return f, ax, PC_dict
    else:
        return PC_dict



def plot_commonplacecells(C_morph_dict,masks,cv_sort=True, plot = True):
    '''plot place place cells across morph values using a cross-validated population sorting. only use place cells
    that have a field in both 0 morph and 1 morph trials
    inputs: C_morph_dict - output from u.trial_type_dict(C_trial_mat, morphs) where C is the [trials, positions,ncells]
                and morphs is [ntrials,] array of mean morph values
            masks - dictionary of place cell masks from place_cells_calc
            cv_sort - bool; calculate sorting from all trials (False) or a randomly selected half of trials (True)
            plot - bool; whether or not to actually generate matplotlib plots or just return sorted data
    outputs: f - figure handle
            ax - axis array
            PC_dict - dictionary of cross-val sorted population activity rate maps'''

    morphs = [k for k in C_morph_dict.keys() if isinstance(k,np.float64)]
    mask = masks[0] & masks[1]
    if plot:
        f,ax = plt.subplots(2,len(morphs),figsize=[5*len(morphs),15])
        f.subplots_adjust(wspace=.01,hspace=.05)

    getSort = lambda fr : np.argsort(np.argmax(np.squeeze(np.nanmean(fr,axis=0)),axis=0))
    PC_dict = {}
    PC_dict[0],PC_dict[1] = {},{}
    if cv_sort:
        ntrials0 = C_morph_dict[0].shape[0]
        sort_trials_0 = np.random.permutation(ntrials0)
        ht0 = int(ntrials0/2)
        arr0 = C_morph_dict[0][:,:,mask]
        arr0 = arr0[sort_trials_0[:ht0],:,:]
        sort0 = getSort(arr0)

        _arr0 = np.copy(arr0)
        _arr0[np.isnan(arr0)]=0.
        norm0 = np.amax(np.nanmean(_arr0,axis=0),axis=0)

        ntrials1 = C_morph_dict[1].shape[0]
        ht1 = int(ntrials1/2)
        sort_trials_1 = np.random.permutation(ntrials1)
        arr1= C_morph_dict[1][:,:,mask]
        arr1 = arr1[sort_trials_1[:ht1],:,:]
        sort1 = getSort(arr1)

        _arr1 = np.copy(arr1)
        _arr1[np.isnan(arr1)]=0.
        norm1= np.amax(np.nanmean(_arr1,axis=0),axis=0)


    else:
        sort0 = getSort(C_morph_dict[0][:,:,masks[0]])
        norm0 = np.amax(np.nanmean(C_morph_dict[0][:,:,masks[0]],axis=0),axis=0)
        sort1 = getSort(C_morph_dict[1][:,:,masks[1]])
        norm1 = np.amax(np.nanmean(C_morph_dict[1][:,:,masks[1]],axis=0),axis=0)



    for i,m in enumerate(morphs):
        if cv_sort:
            if m ==0:
                fr_n0 = np.squeeze(np.nanmean(C_morph_dict[m][sort_trials_0[ht0:],:,:],axis=0))
                fr_n1 = np.squeeze(np.nanmean(C_morph_dict[m],axis=0))
            elif m==1:
                fr_n1 = np.squeeze(np.nanmean(C_morph_dict[m][sort_trials_1[ht1:],:,:],axis=0))
                fr_n0 = np.squeeze(np.nanmean(C_morph_dict[m],axis=0))
            else:
                fr_n0 = np.squeeze(np.nanmean(C_morph_dict[m],axis=0))
                fr_n1 = np.squeeze(np.nanmean(C_morph_dict[m],axis=0))
        else:
            fr_n0 = np.squeeze(np.nanmean(C_morph_dict[m],axis=0))
            fr_n1 = np.squeeze(np.nanmean(C_morph_dict[m],axis=0))

        fr_n0, fr_n1 = fr_n0[:,mask], fr_n1[:,mask]
        fr_n0= gaussian_filter1d(fr_n0/norm0,2,axis=0)
        fr_n1 = gaussian_filter1d(fr_n1/norm1,2,axis=0)
        fr_n0, fr_n1 = fr_n0[:,sort0], fr_n1[:,sort1]

        PC_dict[0][m], PC_dict[1][m]= fr_n0.T, fr_n1.T
        if plot:
            ax[0,i].imshow(fr_n0.T,aspect='auto',cmap='pink',vmin=0.2,vmax=.9)
            ax[1,i].imshow(fr_n1.T,aspect='auto',cmap='pink',vmin=0.2,vmax=.9)
            if i>0:
                ax[0,i].set_yticks([])
                ax[1,i].set_yticks([])
            ax[0,i].set_xticks([])
            ax[1,i].set_xticks([])

    if plot:
        return f, ax, PC_dict
    else:
        return PC_dict
