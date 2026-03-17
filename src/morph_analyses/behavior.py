import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy as sp
import os
from . import utilities as u


def behavior_raster_task(trial_mat, centers, morphs, reward_pos, smooth=True, TO=False):
    '''
    plot trial by position behavioral data as a "smooth raster" for decision making task.
    separate plots are generated sorting trials by actual order, morph value, and reward position
    inputs: trial_mat - [trials, positions] behavioral data to be plotted, can include nans for
                    missing data
            centers - [positins,] position bin centers
            morphs - [trials,] morph values for each trial
            reward_pos - [trials,] position of reward on each trial (scaled 0-1, if greater than 1 assume no reward)
            smooth - bool; whether or not to apply gaussian smoothing across positions
            TO - bool; whether or not position of timeouts should be included for data in
                which that is relevant
    outputs: f - figure handle
            axarr - array of axis handles
    '''

    f = plt.figure(figsize=[15,15])
    gs = gridspec.GridSpec(4,6)
    axarr = []

    # sort by trial order, color by morph
    ax = f.add_subplot(gs[:,:2])
    ax = u.smooth_raster(centers,trial_mat,vals=1-morphs,ax=ax,smooth=smooth,cmap='cool')
    # highlight reward zones
    ax.fill_betweenx([0,trial_mat.shape[0]+1],250,315,color=plt.cm.cool(1.),alpha=.3,zorder=0)
    ax.fill_betweenx([0,trial_mat.shape[0]+1],350,415,color=plt.cm.cool(0.),alpha=.3,zorder=0)
    ax.set_ylabel('Trial',size='xx-large')
    axarr.append(ax)
    if TO: # if data contains timeouts
    # find position of teleports
        to_trial,to_pos = [] ,[]
        for t in range(trial_mat.shape[0]):
            nan_inds = np.isnan(trial_mat[t,:])
            if reward_pos[t]==1:
                to_trial.append(t)
                _pos = centers[nan_inds]
                to_pos.append(_pos[0])
            else:
                to_trial.append(t)
                to_pos.append(np.nan)
        print(to_trial)
        to_pos,to_trial = np.array(to_pos),np.array(to_trial)
        # plot timeouts as red Os
        ax.scatter(to_pos,trial_mat.shape[0]-to_trial-.5,color='red',marker='o')

    # sort trials by morph
    ax = f.add_subplot(gs[:,2:4])
    msort = np.argsort(morphs)
    ax = u.smooth_raster(centers,trial_mat[msort,:],vals=1-morphs[msort],ax=ax,smooth=smooth,cmap='cool')
    ax.fill_betweenx([0,trial_mat.shape[0]+1],250,315,color=plt.cm.cool(1.),alpha=.3,zorder=0)
    ax.fill_betweenx([0,trial_mat.shape[0]+1],350,415,color=plt.cm.cool(0.),alpha=.3,zorder=0)

    if TO:
        ax.scatter(to_pos[msort],trial_mat.shape[0]-np.arange(0,trial_mat.shape[0])-.5,color='red',marker='o')
    axarr.append(ax)



    # sort and color by position of reward
    ax = f.add_subplot(gs[:,4:])
    ax.fill_betweenx([0,trial_mat.shape[0]+1],250,315,color=plt.cm.cool(1.),alpha=.3,zorder=0)
    ax.fill_betweenx([0,trial_mat.shape[0]+1],350,415,color=plt.cm.cool(0.),alpha=.3,zorder=0)
    nor_mask = reward_pos>1
    trial_mat_tmp = np.copy(trial_mat)
    trial_mat_tmp[nor_mask,:]=np.nan
    rsort = np.argsort(reward_pos)
    ax = u.smooth_raster(centers,trial_mat_tmp[rsort,:],vals=reward_pos[rsort],ax=ax,smooth=smooth,cmap='viridis')
    trial_mat_tmp = np.copy(trial_mat)
    trial_mat_tmp[~nor_mask,:]=np.nan
    ax = u.smooth_raster(centers,trial_mat_tmp[rsort,:],vals=np.ones(reward_pos.shape),ax=ax,smooth=smooth,cmap='Greys')
    axarr.append(ax)

    for i,a in enumerate(axarr):
        for edge in ['top','right']:
            a.spines[edge].set_visible(False)
        if i>0:
            a.set_yticklabels([])

    return f, axarr



def behavior_raster_foraging(trial_mat,centers,morphs,reward_pos,smooth=True, rzone=(250,415)):
    '''plot trial by position behavioral data as a "smooth raster" for foraging task.
    separate plots are generated sorting trials by actual order, morph value, and reward position
    inputs: trial_mat - [trials, positions] behavioral data to be plotted, can include nans for
                    missing data
            centers - [positins,] position bin centers
            morphs - [trials,] morph values for each trial
            reward_pos - [trials,] position of reward on each trial (scaled 0-1, if greater than 1 assume no reward)
            smooth - bool; whether or not to apply gaussian smoothing across positions
    outputs: f - figure handle
            axarr - array of axis handles
    '''
    f = plt.figure(figsize=[15,15])
    gs = gridspec.GridSpec(4,6)
    axarr = []

    # sort by trial order, color by morph
    ax = f.add_subplot(gs[:,:2])
    ax = u.smooth_raster(centers,trial_mat,vals=1-morphs,ax=ax,smooth=smooth,cmap='cool')
    # highlight possible reward zone
    ax.fill_betweenx([0,trial_mat.shape[0]+1],rzone[0],rzone[1],color='black',alpha=.3,zorder=0)
    ax.set_ylabel('Trial',size='xx-large')
    axarr.append(ax)


    # sort trials by morph
    ax = f.add_subplot(gs[:,2:4])
    msort = np.argsort(morphs)
    ax = u.smooth_raster(centers,trial_mat[msort,:],vals=1-morphs[msort],ax=ax,smooth=smooth,cmap='cool')
    ax.fill_betweenx([0,trial_mat.shape[0]+1],rzone[0],rzone[1],color='black',alpha=.3,zorder=0)
    axarr.append(ax)



    # sort and color by position of reward
    ax = f.add_subplot(gs[:,4:])
    ax.fill_betweenx([0,trial_mat.shape[0]+1],rzone[0],rzone[1],color='black',alpha=.3,zorder=0)
    nor_mask = reward_pos>1
    trial_mat_tmp = np.copy(trial_mat)
    trial_mat_tmp[nor_mask,:]=np.nan
    rsort = np.argsort(reward_pos)
    ax = u.smooth_raster(centers,trial_mat_tmp[rsort,:],vals=reward_pos[rsort],ax=ax,smooth=smooth,cmap='viridis')
    trial_mat_tmp = np.copy(trial_mat)
    trial_mat_tmp[~nor_mask,:]=np.nan
    ax = u.smooth_raster(centers,trial_mat_tmp[rsort,:],vals=np.ones(reward_pos.shape),ax=ax,smooth=smooth,cmap='Greys')
    axarr.append(ax)

    for i,a in enumerate(axarr):
        for edge in ['top','right']:
            a.spines[edge].set_visible(False)
        if i>0:
            a.set_yticklabels([])

    return f, axarr
