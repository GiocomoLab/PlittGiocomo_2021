import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
import pandas as pd
from datetime import datetime
from glob import glob
from random import randrange
import math
import utilities as u
import preprocessing as pp
import matplotlib.gridspec as gridspec



def cell_topo_plot(A_k,vals,fov=[512,796],map = 'cool', min = -1, max = 1):
    ''' given sparse matrix of cell footprints, A_k, and values associated with
    cells, vals, plot shape of cells colored by vals'''

    nz = A_k.nonzero()
    A= np.zeros(A_k.shape)
    A[nz]=1

    for i, v in enumerate(vals.tolist()):
        A[:,i]*=v

    A_m = np.ma.array(A.max(axis=1) + A.min(axis=1))
    A_m[A_m==0]=np.nan

    f, ax = plt.subplots(figsize=[15,15])
    ax.imshow(np.reshape(A_m,fov,order='F'),cmap=map,vmin=min,vmax=max)

    return A_m, (f,ax)

def plot_top_cells(S_tm,masks,SI,morph,maxcells=400):
    allmask = masks[0]
    for k,v in masks.items():
        allmask = allmask | v

    nplacecells = np.minimum(allmask.sum(),maxcells)

    xstride = 3
    ystride = 4
    nperrow = 8
    f = plt.figure(figsize=[nperrow*xstride,nplacecells/nperrow*ystride])
    gs = gridspec.GridSpec(math.ceil(nplacecells/nperrow)*ystride,xstride*nperrow)

    SI_total = [SI[m]['all'] for m in SI.keys()]
    SIt = SI_total[0]
    for ind in SI_total[1:]:
        SIt+=ind
    si_order = np.argsort(SIt)[::-1]

    morph_order = np.argsort(morph)
    morph_s = morph[morph_order]

    for cell in range(nplacecells): # make this min of 100 and total number of place cells
        c = u.nansmooth(np.squeeze(S_tm[:,:,si_order[cell]]),[0,3])
        c/=np.nanmean(c.ravel())
        # add plots
        row_i = int(ystride*math.floor(cell/nperrow))
        col_i = int(xstride*(cell%nperrow))
        # print(row_i,col_i)
        trialsort_ax = f.add_subplot(gs[row_i:row_i+ystride-1,col_i+1])
        trialsort_ax.imshow(c,cmap='magma',aspect='auto')
        tick_inds = np.arange(0,c.shape[0],10)


        morphsort_ax = f.add_subplot(gs[row_i:row_i+ystride-1,col_i])
        morphsort_ax.imshow(c[morph_order,:],cmap='magma',aspect='auto')
        tick_labels = ["%.2f" % morph_s[i] for i in tick_inds]





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
    f = plt.figure(figsize=[10,10])
    gs = gridspec.GridSpec(5,5)
    ax = f.add_subplot(gs[0:-1,0:-1])


    #f,ax = plt.subplots()
    ax.scatter(5.*np.argmax(fr0,axis=0),5*np.argmax(fr1,axis=0),color='black')
    ax.plot(np.arange(tmax),np.arange(tmax),color='black')
    ax.fill_between(np.arange(tmax),rzone0[0],y2=rzone0[1],color=plt.cm.cool(0),alpha=.2)
    ax.fill_betweenx(np.arange(tmax),rzone0[0],x2=rzone0[1],color=plt.cm.cool(0),alpha=.2)
    ax.fill_between(np.arange(tmax),rzone1[0],y2=rzone1[1],color=plt.cm.cool(1.),alpha=.2)
    ax.fill_betweenx(np.arange(tmax),rzone1[0],x2=rzone1[1],color=plt.cm.cool(1.),alpha=.2)

    ax1 = f.add_subplot(gs[-1,0:-1])
    ax1.hist(5.*np.argmax(fr0,axis=0),np.arange(0,tmax+10,10))
    ax1.fill_betweenx(np.arange(40),rzone0[0],x2=rzone0[1],color=plt.cm.cool(0),alpha=.2)
    ax1.fill_betweenx(np.arange(40),rzone1[0],x2=rzone1[1],color=plt.cm.cool(1.),alpha=.2)

    ax2 = f.add_subplot(gs[0:-1,-1])
    ax2.hist(5.*np.argmax(fr1,axis=0),np.arange(0,tmax+10,10),orientation='horizontal')
    ax2.fill_between(np.arange(40),rzone0[0],y2=rzone0[1],color=plt.cm.cool(0),alpha=.2)
    ax2.fill_between(np.arange(40),rzone1[0],y2=rzone1[1],color=plt.cm.cool(1.),alpha=.2)

    return f, ax


def common_cell_remap_scatterplot(fr0, fr1, rzone = [225,400], tmax= 450,bin_size=10.,norm = True):
    f = plt.figure(figsize=[10,10])
    gs = gridspec.GridSpec(5,5)
    ax = f.add_subplot(gs[0:-1,0:-1])


    heatmap = np.zeros([fr0.shape[0],fr1.shape[0]])
    for cell in range(fr0.shape[1]):
        heatmap[np.argmax(fr0[:,cell],axis=0),np.argmax(fr1[:,cell],axis=0)]+=1
    # print(heatmap/np.amax(heatmap))
    #f,ax = plt.subplots()
    # ax.scatter(bin_size*np.argmax(fr0,axis=0),bin_size*np.argmax(fr1,axis=0),color='black')
    if norm:
        ax.imshow(heatmap.T/np.amax(heatmap),cmap='magma',vmin=0.05,vmax=.15)#,aspect='auto')
    else:
        ax.imshow(heatmap.T,cmap='magma',vmax=.7*np.amax(heatmap.ravel()))
    # ax.plot(np.arange(tmax),np.arange(tmax),color='black')
    # ax.fill_between(np.arange(tmax)/bin_size,rzone[0],y2=rzone[1],color='black',alpha=.2)
    # ax.fill_betweenx(np.arange(tmax)/bin_size,rzone[0],x2=rzone[1],color='black',alpha=.2)


    ax1 = f.add_subplot(gs[-1,0:-1],sharex=ax)
    hist,edges = np.histogram(bin_size*np.argmax(fr0,axis=0),np.arange(0,tmax+10,10))
    ax1.fill_between(np.linspace(0,45,num=edges.shape[0]-1)-.5,hist/hist.sum(),color=plt.cm.cool(1.))
    ax1.set_xlim([-1,46])
    # ax1.hist(bin_size*np.argmax(fr0,axis=0),np.arange(0,tmax+10,10))
    ax1.fill_betweenx([0,.05],rzone[0]/bin_size,x2=rzone[1]/bin_size,color='black',alpha=.2)


    ax2 = f.add_subplot(gs[0:-1,-1],sharey=ax)
    hist,edges = np.histogram(bin_size*np.argmax(fr1,axis=0),np.arange(0,tmax+10,10))
    ax2.fill_betweenx(np.linspace(0,45,num=edges.shape[0]-1)-.5,hist/hist.sum(),color=plt.cm.cool(0.))
    ax2.set_ylim([-1,46])
    # ax2.hist(bin_size*np.argmax(fr1,axis=0),np.arange(0,tmax+10,10),orientation='horizontal',color='black')
    ax2.fill_between([0,.05],rzone[0]/bin_size,y2=rzone[1]/bin_size,color='black',alpha=.2)


    return f, ax


def stability_split_halves(trial_mat):
    '''calculate first half vs second half tuning curve correlation'''

    # assume trial_mat is (trials x pos x cells)
    half = int(trial_mat.shape[0]/2)

    fr0 = np.squeeze(np.nanmean(trial_mat[:half,:,:],axis=0))
    fr1 = np.squeeze(np.nanmean(trial_mat[half:,:,:],axis=0))

    sc_corr, pv_corr = stability(fr0,fr1)
    return sc_corr, pv_corr

def stability(fr0, fr1):
    # single cell place cell correlations
    sc_corr = np.array([sp.stats.pearsonr(fr0[:,cell],fr1[:,cell]) for cell in range(fr0.shape[1])])

    # population vector correlation
    pv_corr  = np.array([sp.stats.pearsonr(fr0[pos,:],fr1[pos,:]) for pos in range(fr0.shape[0])])
    return sc_corr, pv_corr

def meanvectorlength(fr):
    return np.linalg.norm(fr-fr.mean())

def spatial_info(frmap,occupancy):
    '''calculate spatial information bits/spike'''
    ncells = frmap.shape[1]

    SI = []
    ### vectorizing
    P_map = frmap - np.amin(frmap)+.001
    # P_map = gaussian_filter(P_map,[3,0])
    P_map = P_map/P_map.mean(axis=0)
    arg = P_map*occupancy[:,np.newaxis]
    denom = arg.sum(axis=0)
    # SI = (arg*np.log2(P_map/denom)).sum(axis=0)
    SI = (arg*np.log2(P_map)).sum(axis=0)
    # SI = (P_map*occupancy[:,np.newaxis]*np.log2(P_map)).sum(axis=0)
    return SI



def place_cells_calc(C, position, trial_info, tstart_inds,
                teleport_inds,method="all",pthr = .99,correct_only=False,
                speed=None,win_trial_perm=False,morphlist = [0,1]):
    '''get masks for significant place cells that have significant place info
    in both even and odd trials'''


    C_trial_mat, occ_trial_mat, edges,centers = u.make_pos_bin_trial_matrices(C,position,tstart_inds,teleport_inds,speed = speed)

    morphs = trial_info['morphs']
    if correct_only:
        mask = trial_info['rewards']>0
        morphs = morphs[mask]
        C_trial_mat = C_trial_mat[mask,:,:]
        occ_trial_mat = occ_trial_mat[mask,:]

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
        occ_all = occ_morph_dict[m].sum(axis=0)
        occ_all /= occ_all.sum()
        SI[m]['all'] =  spatial_info(FR[m]['all'],occ_all)
        if method == 'split_halves':
            FR[m]['odd'] = np.nanmean(C_morph_dict[m][0::2,:,:],axis=0)
            FR[m]['even'] = np.nanmean(C_morph_dict[m][1::2,:,:],axis=0)

            # occupancy
            occ_o, occ_e = occ_morph_dict[m][0::2,:].sum(axis=0), occ_morph_dict[m][1::2,:].sum(axis=0)
            occ_o/=occ_o.sum()
            occ_e/=occ_e.sum()



            SI[m]['odd'] = spatial_info(FR[m]['odd'],occ_o)
            SI[m]['even'] = spatial_info(FR[m]['even'],occ_e)


            p_e, shuffled_SI = spatial_info_perm_test(SI[m]['even'],C,position,
                                    tstart_morph_dict[m][1::2],teleport_morph_dict[m][1::2],
                                    nperms=1000,win_trial=win_trial_perm)
            p_o, shuffled_SI = spatial_info_perm_test(SI[m]['odd'],C,position,
                                    tstart_morph_dict[m][0::2],teleport_morph_dict[m][0::2],
                                    nperms = 1000,win_trial=win_trial_perm ) #,shuffled_SI=shuffled_SI)


            masks[m]=np.multiply(p_e>pthr,p_o>pthr)

        elif method == 'bootstrap':
            n_boots=30
            # drop trial with highest firing rate
            tmat = C_morph_dict[m]
            # maxtrial = np.argmax(tmat.sum(axis=1),axis=0)
            # print(maxtrial.shape)
            # mask = np.ones([tmat.shape[0],])
            # mask[maxtrial]=0
            # mask = mask>0
            # tmat = tmat[mask,:,:]
            omat = occ_morph_dict[m]#[mask,:,:]

            SI_bs = np.zeros([n_boots,C.shape[1]])
            print("start bootstrap")
            for b in range(n_boots):

                # pick a random subset of trials
                ntrials = tmat.shape[0] #C_morph_dict[m].shape[0]
                bs_pcnt = .67 # proportion of trials to keep
                bs_thr = int(bs_pcnt*ntrials) # number of trials to keep
                bs_inds = np.random.permutation(ntrials)[:bs_thr]
                FR_bs = np.nanmean(tmat[bs_inds,:,:],axis=0)
                    #np.nanmean(C_morph_dict[m][bs_inds,:,:],axis=0)
                occ_bs = omat[bs_inds,:].sum(axis=0)#occ_morph_dict[m][bs_inds,:].sum(axis=0)
                occ_bs/=occ_bs.sum()
                SI_bs[b,:] = spatial_info(FR_bs,occ_bs)
            print("end bootstrap")
            SI[m]['bootstrap']= np.median(SI_bs,axis=0).ravel()
            p_bs, shuffled_SI = spatial_info_perm_test(SI[m]['bootstrap'],C,
                                    position,tstart_morph_dict[m],teleport_morph_dict[m],
                                    nperms=100,win_trial=win_trial_perm)
            masks[m] = p_bs>pthr

        else:
            p_all, shuffled_SI = spatial_info_perm_test(SI[m]['all'],C,position,
                                    tstart_morph_dict[m],teleport_morph_dict[m],
                                    nperms=1000,win_trial=win_trial_perm)
            masks[m] = p_all>pthr

    return masks, FR, SI



def spatial_info_perm_test(SI,C,position,tstart,tstop,nperms = 10000,shuffled_SI=None,win_trial = True):
    '''run permutation test on spatial information calculations. returns empirical p-values for each cell'''
    if len(C.shape)>2:
        C = np.expand_dims(C,1)

    if shuffled_SI is None:
        shuffled_SI = np.zeros([nperms,C.shape[1]])

        for perm in range(nperms):

            if win_trial:
                C_tmat, occ_tmat, edes,centers = u.make_pos_bin_trial_matrices(C,position,tstart,tstop,perm=True)
            else:
                C_perm = np.roll(C,randrange(30,position.shape[0],30),axis=0)
                C_tmat, occ_tmat, edes,centers = u.make_pos_bin_trial_matrices(C,position,tstart,tstop,perm=False)

            fr, occ = np.squeeze(np.nanmean(C_tmat,axis=0)), occ_tmat.sum(axis=0)
            occ/=occ.sum()

            si = spatial_info(fr,occ)
            shuffled_SI[perm,:] = si


    p = np.zeros([C.shape[1],])
    for cell in range(C.shape[1]):
        #print(SI[cell],np.max(shuffled_SI[:,cell]))
        #p[cell] = np.where(SI[cell]>shuffled_SI[:,cell])[0].shape[0]/nperms
        p[cell] = np.sum(SI[cell]>shuffled_SI[:,cell])/nperms

    return p, shuffled_SI

def plot_placecells(C_morph_dict,masks,cv_sort=True, plot = True):
    '''plot place place cell results'''

    morphs = [k for k in C_morph_dict.keys() if isinstance(k,np.float64)]
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
        arr0 = C_morph_dict[0][:,:,masks[0]]
        arr0 = arr0[sort_trials_0[:ht0],:,:]
        sort0 = getSort(arr0)

        _arr0 = np.copy(arr0)
        _arr0[np.isnan(arr0)]=0.
        norm0 = np.amax(np.nanmean(_arr0,axis=0),axis=0)

        ntrials1 = C_morph_dict[1].shape[0]
        ht1 = int(ntrials1/2)
        sort_trials_1 = np.random.permutation(ntrials1)
        arr1= C_morph_dict[1][:,:,masks[1]]
        arr1 = arr1[sort_trials_1[:ht1],:,:]
        sort1 = getSort(arr1)

        _arr1 = np.copy(arr1)
        _arr1[np.isnan(arr1)]=0.
        norm1= np.amax(np.nanmean(_arr1,axis=0),axis=0)


    else:
        sort0 = getSort(C_morph_dict[0][:,:,masks[0]])
        sort1 = getSort(C_morph_dict[1][:,:,masks[1]])


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

        fr_n0, fr_n1 = fr_n0[:,masks[0]], fr_n1[:,masks[1]]
        fr_n0= gaussian_filter1d(fr_n0/norm0,2,axis=0)
        fr_n1 = gaussian_filter1d(fr_n1/norm1,2,axis=0)
        # for j in range(fr_n0.shape[1]):
            # fr_n0[:,j] = gaussian_filter1d(fr_n0[:,j]/norm0,2) #/fr_n0[:,j].mean(),2)
            # fr_n1[:,j] = gaussian_filter1d(fr_n1[:,j]/norm1,2) #/fr_n1[:,j].mean(),2)
            #fr_n[:,j] = gaussian_filter1d(fr[:,j],2)

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
    '''plot place place cell results'''

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
        sort0 = getSort(C_morph_dict[0][:,:,mask])
        sort1 = getSort(C_morph_dict[1][:,:,mask])


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
        # for j in range(fr_n0.shape[1]):
            # fr_n0[:,j] = gaussian_filter1d(fr_n0[:,j]/norm0,2) #/fr_n0[:,j].mean(),2)
            # fr_n1[:,j] = gaussian_filter1d(fr_n1[:,j]/norm1,2) #/fr_n1[:,j].mean(),2)
            #fr_n[:,j] = gaussian_filter1d(fr[:,j],2)

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
