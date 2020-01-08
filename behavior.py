import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy as sp
import os
import utilities as u


def behavior_raster_task(lick_mat,centers,morphs,reward_pos,smooth=True, max_pos=None,TO=False):
    '''plot licking or running behavior when the animal is doing the 'foraging' task '''

    f = plt.figure(figsize=[15,15])
    gs = gridspec.GridSpec(4,6)
    axarr = []

    # lick x pos - colored by morph
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
            a.set_yticklabels([])

    return f, axarr
