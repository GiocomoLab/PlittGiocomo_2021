import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy as sp
import os
from astropy.convolution import convolve, Gaussian1DKernel
import utilities as u



def lick_plot_task(d,bin_edges,rzone0=(250.,315),rzone1=(350,415),smooth=True,ratio = True, max_pos=None):
    '''standard plot for licking behavior'''
    f = plt.figure(figsize=[15,15])

    gs = gridspec.GridSpec(5,5)


    ax = f.add_subplot(gs[0:-1,0:-1])
    ax.axvspan(rzone0[0],rzone0[1],alpha=.2,color=plt.cm.cool(np.float(1.)),zorder=0)
    ax.axvspan(rzone1[0],rzone1[1],alpha=.2,color=plt.cm.cool(np.float(0.)),zorder=0)
    ax = u.smooth_raster(bin_edges[:-1],d['all'],vals=1-d['labels'],ax=ax,smooth=smooth,tports=max_pos)
    ax.set_ylabel('Trial',size='xx-large')


    meanlr_ax = f.add_subplot(gs[-1,:-1])
    meanlr_ax.axvspan(rzone0[0],rzone0[1],alpha=.2,color=plt.cm.cool(np.float(1)),zorder=0)
    meanlr_ax.axvspan(rzone1[0],rzone1[1],alpha=.2,color=plt.cm.cool(np.float(0)),zorder=0)
    for i, m in enumerate(np.unique(d['labels'])):
        meanlr_ax.plot(bin_edges[:-1],np.nanmean(d[m],axis=0),color=plt.cm.cool(1-np.float(m)))
    meanlr_ax.set_ylabel('Licks/sec',size='xx-large')
    meanlr_ax.set_xlabel('Position (cm)',size='xx-large')


    if ratio:
        lickrat_ax = f.add_subplot(gs[:-1,-1])
        bin_edges = np.array(bin_edges)
        rzone0_inds = np.where((bin_edges[:-1]>=rzone0[0]) & (bin_edges[:-1] <= rzone0[1]))[0]
        rzone1_inds = np.where((bin_edges[:-1]>=rzone1[0]) & (bin_edges[:-1] <= rzone1[1]))[0]
        rzone_lick_ratio = {}
        for i,m in enumerate(np.unique(d['labels'])):
            zone0_lick_rate = d[m][:,rzone0_inds].mean(axis=1)
            zone1_lick_rate = d[m][:,rzone1_inds].mean(axis=1)
            rzone_lick_ratio[m] = np.divide(zone0_lick_rate,zone0_lick_rate+zone1_lick_rate)
            rzone_lick_ratio[m][np.isinf(rzone_lick_ratio[m])]=np.nan

        for i,m in enumerate(np.unique(d['labels'])):

            trial_index = d['labels'].shape[0] - d['indices'][m]
            lickrat_ax.scatter(rzone_lick_ratio[m],trial_index,
                               c=plt.cm.cool(np.float(m)),s=10)
            k = Gaussian1DKernel(5)
            lickrat_ax.plot(convolve(rzone_lick_ratio[m],k,boundary='extend'),trial_index,c=plt.cm.cool(np.float(m)))
        lickrat_ax.set_yticklabels([])
        lickrat_ax.set_xlabel(r'$\frac{zone_0}{zone_0 + zone_1}  $',size='xx-large')


        for axis in [ax, meanlr_ax, lickrat_ax]:
            for edge in ['top','right']:
                axis.spines[edge].set_visible(False)

        return f, (ax, meanlr_ax, lickrat_ax)
    else:
        for axis in [ax, meanlr_ax]:
            for edge in ['top','right']:
                axis.spines[edge].set_visible(False)

        return f, (ax, meanlr_ax)


def behavior_raster_task(lick_mat,centers,morphs,reward_pos,smooth=True, max_pos=None,TO=False):
    '''plot licking or running behavior when the animal is doing the 'foraging' task '''

    f = plt.figure(figsize=[15,15])
    gs = gridspec.GridSpec(4,6)
    axarr = []

    # lick x pos
    #       colored by morph
    ax = f.add_subplot(gs[:,:2])
    ax = u.smooth_raster(centers,lick_mat,vals=1-morphs,ax=ax,smooth=smooth,cmap='cool')
    ax.fill_betweenx([0,lick_mat.shape[0]+1],250,315,color=plt.cm.cool(1.),alpha=.3,zorder=0)
    ax.fill_betweenx([0,lick_mat.shape[0]+1],350,415,color=plt.cm.cool(0.),alpha=.3,zorder=0)
    ax.set_ylabel('Trial',size='xx-large')
    axarr.append(ax)
    if TO:
        to_trial,to_pos = [] ,[]
        for t in range(lick_mat.shape[0]):
            nan_inds = np.isnan(lick_mat[t,:])
            if reward_pos[t]==1:
                to_trial.append(t)
                _pos = centers[nan_inds]
                to_pos.append(_pos[0])
            else:
                to_trial.append(t)
                to_pos.append(np.nan)
        print(to_trial)
        to_pos,to_trial = np.array(to_pos),np.array(to_trial)
        ax.scatter(to_pos,lick_mat.shape[0]-to_trial-.5,color='red',marker='o')

    # sort trials by morph
    ax = f.add_subplot(gs[:,2:4])
    msort = np.argsort(morphs)
    ax = u.smooth_raster(centers,lick_mat[msort,:],vals=1-morphs[msort],ax=ax,smooth=smooth,cmap='cool')
    ax.fill_betweenx([0,lick_mat.shape[0]+1],250,315,color=plt.cm.cool(1.),alpha=.3,zorder=0)
    ax.fill_betweenx([0,lick_mat.shape[0]+1],350,415,color=plt.cm.cool(0.),alpha=.3,zorder=0)

    if TO:
        ax.scatter(to_pos[msort],lick_mat.shape[0]-np.arange(0,lick_mat.shape[0])-.5,color='red',marker='o')

    axarr.append(ax)



    #       colored by position of reward

    ax = f.add_subplot(gs[:,4:])
    # ax.axvline(200,ymin=0,ymax=lick_mat.shape[0]+1)
    ax.fill_betweenx([0,lick_mat.shape[0]+1],250,315,color=plt.cm.cool(1.),alpha=.3,zorder=0)
    ax.fill_betweenx([0,lick_mat.shape[0]+1],350,415,color=plt.cm.cool(0.),alpha=.3,zorder=0)
    nor_mask = reward_pos>1
    lick_mat_tmp = np.copy(lick_mat)
    lick_mat_tmp[nor_mask,:]=np.nan
    rsort = np.argsort(reward_pos)
    ax = u.smooth_raster(centers,lick_mat_tmp[rsort,:],vals=reward_pos[rsort],ax=ax,smooth=smooth,cmap='viridis')
    lick_mat_tmp = np.copy(lick_mat)
    lick_mat_tmp[~nor_mask,:]=np.nan
    ax = u.smooth_raster(centers,lick_mat_tmp[rsort,:],vals=np.ones(reward_pos.shape),ax=ax,smooth=smooth,cmap='Greys')
    # ax = u.smooth_raster(centers,)
    axarr.append(ax)

    for i,a in enumerate(axarr):
        for edge in ['top','right']:
            a.spines[edge].set_visible(False)
        if i>0:
            # a.spines['left'].set_visible(False)
            a.set_yticklabels([])




    return f, axarr



def behavior_raster_foraging(lick_mat,centers,morphs,reward_pos,smooth=True, max_pos=None,rzone=(250,415)):
    '''plot licking or running behavior when the animal is doing the 'foraging' task '''

    f = plt.figure(figsize=[15,15])
    gs = gridspec.GridSpec(4,6)
    axarr = []

    # lick x pos
    #       colored by morph
    ax = f.add_subplot(gs[:,:2])
    ax = u.smooth_raster(centers,lick_mat,vals=1-morphs,ax=ax,smooth=smooth,cmap='cool')
    ax.fill_betweenx([0,lick_mat.shape[0]+1],225,400,color='black',alpha=.3,zorder=0)
    ax.set_ylabel('Trial',size='xx-large')
    axarr.append(ax)


    # sort trials by morph
    ax = f.add_subplot(gs[:,2:4])
    msort = np.argsort(morphs)
    ax = u.smooth_raster(centers,lick_mat[msort,:],vals=1-morphs[msort],ax=ax,smooth=smooth,cmap='cool')
    ax.fill_betweenx([0,lick_mat.shape[0]+1],225,400,color='black',alpha=.3,zorder=0)
    axarr.append(ax)



    #       colored by position of reward

    ax = f.add_subplot(gs[:,4:])
    # ax.axvline(200,ymin=0,ymax=lick_mat.shape[0]+1)
    ax.fill_betweenx([0,lick_mat.shape[0]+1],225,400,color='black',alpha=.3,zorder=0)
    nor_mask = reward_pos>1
    lick_mat_tmp = np.copy(lick_mat)
    lick_mat_tmp[nor_mask,:]=np.nan
    rsort = np.argsort(reward_pos)
    ax = u.smooth_raster(centers,lick_mat_tmp[rsort,:],vals=reward_pos[rsort],ax=ax,smooth=smooth,cmap='viridis')
    lick_mat_tmp = np.copy(lick_mat)
    lick_mat_tmp[~nor_mask,:]=np.nan
    ax = u.smooth_raster(centers,lick_mat_tmp[rsort,:],vals=np.ones(reward_pos.shape),ax=ax,smooth=smooth,cmap='Greys')
    # ax = u.smooth_raster(centers,)
    axarr.append(ax)

    for i,a in enumerate(axarr):
        for edge in ['top','right']:
            a.spines[edge].set_visible(False)
        if i>0:
            # a.spines['left'].set_visible(False)
            a.set_yticklabels([])




    return f, axarr


def ant_speed_v_lick(lick_mat,speed_mat,centers,morphs):
    ''''''

    f,ax = plt.subplots()

    mask = (centers>=200) & (centers<=250)

    ax.scatter(lick_mat[:,mask].sum(axis=1),speed_mat[:,mask].mean(axis=1),c=1-morphs,cmap='cool')
    return f,ax
