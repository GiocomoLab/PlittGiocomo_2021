import pdb
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from scipy.ndimage import filters
from astropy.convolution import convolve, Gaussian1DKernel
import sklearn as sk
from sklearn import neighbors


### useful general purpose functions for data analysis ###

def gaussian(mu,sigma,x):
    '''radial basis function centered at 'mu' with width 'sigma', sampled at 'x' '''
    return np.exp(-(mu-x)**2/sigma**2)


# def generalized_similarity_fraction(S_trial_mat,morphs,s = np.linspace(-.1,1.1,num=50),argmax=True):
#     S_trial_mat = sp.ndimage.filters.gaussian_filter1d(S_trial_mat,1,axis=1) # smooth position by 1 bin
#
#     #flatten to be trial x positions*neurons
#     S_tmat = np.reshape(S_trial_mat,[S_trial_mat.shape[0],-1])
#      # divide trials by l2-norm
#     S_tmat_norm = S_tmat/np.linalg.norm(S_tmat,ord=2,axis=-1)[:,np.newaxis]
#
#     gsf = np.zeros((morphs.shape[0],s.shape[0])) # similarity fraction
#     for trial in range(morphs.shape[0]): # for each trial
#         trainmask = np.ones((S_tmat.shape[0],))
#         trainmask[trial]=0
#         trainmask = trainmask>0
#
#         # fit NN regressor
#         knnr = sk.neighbors.KNeighborsRegressor(n_neighbors=10)
#         knnr.fit(morphs[trainmask,np.newaxis],S_tmat[trainmask,:])
#
#         centroids = knnr.predict(s[:,np.newaxis])
#         centroids = centroids/np.linalg.norm(centroids,ord=2,axis=1,keepdims=True)
#
#         gsf[trial,:] = np.dot(centroids,S_tmat_norm[trial,:].T)
#
#     gsf = gsf/gsf.sum(axis=1,keepdims=True)
#     if argmax:
#         return s[np.argmax(gsf,axis=1)]
#     else:
#         return gsf

def similarity_fraction(S_trial_mat,trial_info):
    '''
    calculate cosine similarity to average morph=1 represenation divided by the sum of cosine similarities to the two extremes
    similar to a coding direction but on unit circle
    inputs: S_trial_mat - [trials, position bins, neurons] numpy array of activity rates
            trial_info - dictionary of trial information. output of by_trial_info
    returns: sf - [trials,] numpy array of similarity fraction 
    '''
    S_trial_mat = sp.ndimage.filters.gaussian_filter1d(S_trial_mat,2,axis=1) # smooth position by 1 bin

    #flatten to be trial x positions*neurons
    S_tmat = np.reshape(S_trial_mat,[S_trial_mat.shape[0],-1])
     # divide trials by l2-norm
    S_tmat_norm = S_tmat/np.linalg.norm(S_tmat,ord=2,axis=-1)[:,np.newaxis]

    sf = np.zeros(trial_info['morphs'].shape[0]) # similarity fraction
    for trial in range(trial_info['morphs'].shape[0]): # for each trial
        # get masks for centroids
        mask0 = trial_info['morphs']==0
        mask1 = trial_info['morphs']==1
        # if current trial is in mask, exclude it
        if trial_info['morphs'][trial]==0:
            mask0[trial]=False
        elif trial_info['morphs'][trial]==1:
            mask1[trial]=False

        # calculate centroids
        centroid0, centroid1 = np.nanmean(S_tmat_norm[mask0,:],axis=0), np.nanmean(S_tmat_norm[mask1,:],axis=0)
        # cd = centroid1 - centroid0
        # cd = cd/np.linalg.norm(cd,ord=2)
        centroid0=centroid0/np.linalg.norm(centroid0,ord=2)
        centroid1=centroid1/np.linalg.norm(centroid1,ord=2)



        # similarity to two centroids
        angle0,angle1 = np.dot(S_tmat_norm[trial,:],centroid0),np.dot(S_tmat_norm[trial,:],centroid1)
        # whole trial similarity fraction
        sf[trial] = angle1/(angle0+angle1)
        # sf[trial] = np.dot(cd,S_tmat[trial,:])
    return sf


def _first_sess_gen(mlist,fs, default_first = 5):
    '''create list of first sessions to allow flexible inputs in other functions
    inputs: mlist - list of mice
            fs - first_sessions. can be list, int, or None
    returns: fs - list of first sessions'''
    if fs is None:
        return len(mlist)*[default_first]
    elif isinstance(fs,int):
        return len(mlist)*[fs]
    else:
        return fs

class LOTrialO:
    '''Iterator for train and test indices for  leave-one-trial-out cross-validation
    using all timepoints.
    usage: for train,test in LOTrialO(starts,stops,S.shape[0]):
                S_train,S_test = S[train,:], S[test,:]
                model.fit(S_train)
                model.test(S_test)

            starts - indices of trial starts
            stops - indices of trial stops
            N - size of timepoints dimension

            returns: train - boolean of training indices
                    test - boolean of test indices
            '''
    def __init__(self,starts,stops,N):
        self.train_mask = np.zeros([N,])
        self.test_mask = np.zeros([N,])
        self.starts = starts
        self.stops = stops
        self.N = N

    def __iter__(self):
        self.c=-1
        return self

    def get_masks(self):
        self.train_mask *= 0
        self.test_mask *= 0
        for t,(start,stop) in enumerate(zip(self.starts,self.stops)):
            if t == self.c:
                self.test_mask[start:stop]+=1
            else:
                self.train_mask[start:stop]+=1
        return self.train_mask>0,self.test_mask>0

    def __next__(self):
        self.c+=1
        if self.c>=self.starts.shape[0]:
            raise StopIteration
        train,test = self.get_masks()
        return train, test

def nansmooth(A,sig):
    '''apply Gaussian smoothing to matrix A containing nans with kernel sig
    without propogating nans'''

    #find nans
    nan_inds = np.isnan(A)
    A_nanless = np.copy(A)
    # make nans 0
    A_nanless[nan_inds]=0

    # inversely weight nanned indices
    One = np.ones(A.shape)
    One[nan_inds]=.001
    A_nanless= filters.gaussian_filter(A_nanless,sig)
    One = filters.gaussian_filter(One,sig)
    return A_nanless/One

def df(C,ops={'sig_baseline':10,'win_baseline':300,'sig_output':3,'method':'maximin'}):
    '''delta F / F using maximin method from Suite2P
    inputs: C - neuropil subtracted fluorescence (timepoints x neurons)
    outputs dFF - timepoints x neurons'''
    if ops['method']=='maximin': # windowed baseline estimation
        Flow = filters.gaussian_filter(C,    [ops['sig_baseline'], 0])
        Flow = filters.minimum_filter1d(Flow,    ops['win_baseline'],axis=0)
        Flow = filters.maximum_filter1d(Flow,    ops['win_baseline'],axis=0)
    else:
        pass

    C-=Flow # substract baseline (dF)
    C/=Flow # divide by baseline (dF/F)
    return filters.gaussian_filter(C,[ops['sig_output'],0]) # smooth result

def correct_trial_mask(rewards,starts,stops,N):
    '''create mask for indices where rewards is greater than 0
    inputs: rewards - [trials,] list or array with number of rewards per trial
            starts - list of indices for trial starts
            stops - list of inidices for trial stops
            N - length of total timeseries (i.e. S.shape[0])
    outputs: pcnt - mask of indices for trials where the animal received a reward'''
    pcnt = np.zeros([N,]) # initialize

    # loop through trials and make mask
    for i,(start,stop) in enumerate(zip(starts,stops)):
        pcnt[start:stop] = int(rewards[i]>0)
    return pcnt



def lick_positions(licks,position):
    ''' creates vector of lick positions for making a lick raster
    inputs: licks - [timepoints,] or [timepoints,1] vector of number of licks at each timepoint
            positions - corresponding vector of positions
    outputs: lickpos - nans where licks==0 and position where licks>0'''

    lickpos = np.zeros([licks.shape[0],])
    lickpos[:]=np.nan
    lick_inds = np.where(licks>0)[0]
    lickpos[lick_inds]=position[lick_inds]
    return lickpos


def make_pos_bin_trial_matrices(arr, pos, tstart_inds, tstop_inds,bin_size=5,
                                max_pos=450,speed=None,speed_thr=2, perm=False,
                                mat_only = False):
    '''make a ntrials x position [x neurons] matrix[/tensor]---heavily used
    inputs: arr - timepoints x anything array to be put into trials x positions format
            pos - position at each timepoint
            tstart_inds - indices of trial starts
            tstop_inds - indices of trial stops
            bin_size - spatial bin size in cm
            max_pos - maximum position on track
            speed - vector of speeds at each timepoint. If None, then no speed filtering is done
            speed_thr - speed threshold in cm/s. Timepoints of low speed are dropped
            perm - bool. whether to circularly permute timeseries before binning. used for permutation testing
            mat_only - bool. return just spatial binned data or also occupancy, bin edges, and bin bin_centers

    outputs: if mat_only
                    trial_mat - position binned data
            else
                    trial_mat
                    occ_mat - trials x positions matrix of bin occupancy
                    bin_edges - position bin edges
                    bin_centers - bin centers '''


    ntrials = tstart_inds.shape[0]
    if speed is not None: # mask out speeds below speed threshold
        pos[speed<speed_thr]=-1000
        arr[speed<speed_thr,:]=np.nan

    # make position bins
    bin_edges = np.arange(0,max_pos+bin_size,bin_size)
    bin_centers = bin_edges[:-1]+bin_size/2
    bin_edges = bin_edges.tolist()


    # if arr is a vector, expand dimension
    if len(arr.shape)<2:
        arr = np.expand_dims(arr,axis=1)


    trial_mat = np.zeros([int(ntrials),len(bin_edges)-1,arr.shape[1]])
    trial_mat[:] = np.nan
    occ_mat = np.zeros([int(ntrials),len(bin_edges)-1])
    for trial in range(int(ntrials)): # for each trial
            # get trial indices
            firstI, lastI = tstart_inds[trial], tstop_inds[trial]

            arr_t,pos_t = arr[firstI:lastI,:], pos[firstI:lastI]
            if perm: # circularly permute if desired
                pos_t = np.roll(pos_t,np.random.randint(pos_t.shape[0]))

            # average within spatial bins
            for b, (edge1,edge2) in enumerate(zip(bin_edges[:-1],bin_edges[1:])):
                if np.where((pos_t>edge1) & (pos_t<=edge2))[0].shape[0]>0:
                    trial_mat[trial,b] = np.nanmean(arr_t[(pos_t>edge1) & (pos_t<=edge2),:],axis=0)
                    occ_mat[trial,b] = np.where((pos_t>edge1) & (pos_t<=edge2))[0].shape[0]
                else:
                    pass

    if mat_only:
        return np.squeeze(trial_mat)
    else:
        return np.squeeze(trial_mat), np.squeeze(occ_mat/occ_mat.sum(axis=1)[:,np.newaxis]), bin_edges, bin_centers



def trial_type_dict(mat,type_vec):
    '''make dictionary where each key is a trial type and data is arbitrary trial x var x var data
    should be robust to whether or not non-trial dimensions exist
    inputs: mat - data to be put in dictionary
            type_vec - [# trials, ] vector of labels used for making dictionary
    outputs: d - dictionary or split data
    '''

    d = {'all': np.squeeze(mat)}
    ndim = len(d['all'].shape)
    d['labels'] = type_vec
    d['indices']={}
    for i,m in enumerate(np.unique(type_vec)):
        d['indices'][m] = np.where(type_vec==m)[0]

        if ndim==1:
            d[m] = d['all'][d['indices'][m]]
        elif ndim==2:
            d[m] = d['all'][d['indices'][m],:]
        elif ndim==3:
            d[m] = d['all'][d['indices'][m],:,:]
        else:
            raise(Exception("trial matrix is incorrect dimensions"))

    return d



def by_trial_info(data,rzone0=(250,315),rzone1=(350,415)):
    '''get relevant single trial behavioral information and return a dictionary
    inputs: data - VRDat pandas dataframe from preprocessing.behavior_dataframe
            rzone0 - reward zone for S=0 morph for context discrimination task
            rzone1 - reward zone for S=1 morph for context discrimination task
    outputs: trial_info - dictionary of single trial information
            tstart_inds - array of trial starts
            teleport_inds - array of trial stops
    '''


    # find trial start and stops
    tstart_inds, teleport_inds = data.index[data.tstart==1],data.index[data.teleport==1]


    trial_info={}
    morphs = np.zeros([tstart_inds.shape[0],]) # mean morph
    max_pos = np.zeros([tstart_inds.shape[0],]) # maximum position reached by the animal
    rewards = np.zeros([tstart_inds.shape[0],]) # number of rewards dispensed on trial
    wallJitter= np.zeros([tstart_inds.shape[0],]) # jitter added to wall cues
    towerJitter= np.zeros([tstart_inds.shape[0],]) # jitter added to tower color
    bckgndJitter= np.zeros([tstart_inds.shape[0],]) # jitter added to background color
    clickOn= np.zeros([tstart_inds.shape[0],]) # autoreward on?
    pos_lick = np.nan*np.zeros([tstart_inds.shape[0],]) # posisition of first lick
    # pos_lick[:] = np.nan

    # discrimination task
    omissions = np.zeros([tstart_inds.shape[0],]) # whether the animal missed the reward zone
    zone0_licks = np.zeros([tstart_inds.shape[0],]) # licks in reward zone 0
    zone1_licks = np.zeros([tstart_inds.shape[0],]) # licks in reward zone 1
    zone0_speed = np.zeros([tstart_inds.shape[0],]) # speed in reward zone 0
    zone1_speed = np.zeros([tstart_inds.shape[0],]) # speed in reward zone 1
    pcnt = np.nan*np.zeros([tstart_inds.shape[0],]) # for psychometric curve plotting

    # foraging task info (and discrimination task)
    reward_pos = np.nan*np.zeros([tstart_inds.shape[0],]) # position reward was delivered
    rzone_entry = np.nan*np.zeros([tstart_inds.shape[0],]) # position animal entered reward zone


    for (i,(s,f)) in enumerate(zip(tstart_inds,teleport_inds)): # for each trial

        sub_frame = data[s:f] # get rows for trial

        # get the morph value for that trial. omit if it's undefined
        m, counts = sp.stats.mode(sub_frame['morph'],nan_policy='omit')
        if len(m)>0:
            morphs[i] = m
            max_pos[i] = np.nanmax(sub_frame['pos'])
            rewards[i] = np.nansum(sub_frame['reward'])

            # if reward was delivered
            if rewards[i]>0:
                # position of reward
                rpos=sub_frame.loc[sub_frame['reward']>0,'pos']
                reward_pos[i]=rpos._values[0]

                # entry to reward zone
                rzone_poss = sub_frame.loc[sub_frame['rzone']>0,'pos']
                if rzone_poss.shape[0]>0:
                    rzone_entry[i]=rzone_poss._values[0]
                else:
                    rzone_entry[i]=reward_pos[i]


            wj, c = sp.stats.mode(sub_frame['wallJitter'],nan_policy='omit')
            wallJitter[i] = wj
            tj, c = sp.stats.mode(sub_frame['towerJitter'],nan_policy='omit')
            towerJitter[i] = tj
            bj, c = sp.stats.mode(sub_frame['bckgndJitter'],nan_policy='omit')
            bckgndJitter[i] = bj
            co, c = sp.stats.mode(sub_frame['clickOn'],nan_policy='omit')
            clickOn[i]=co


            ### discrimination task stuff
            zone0_mask = (sub_frame.pos>=rzone0[0]) & (sub_frame.pos<=rzone0[1])
            zone1_mask = (sub_frame.pos>=rzone1[0]) & (sub_frame.pos<=rzone1[1])
            zone0_licks[i] = np.nansum(sub_frame.loc[zone0_mask,'lick'])
            zone1_licks[i] = np.nansum(sub_frame.loc[zone1_mask,'lick'])
            zone0_speed[i]=np.nanmean(sub_frame.loc[zone0_mask,'speed'])
            zone1_speed[i] = np.nanmean(sub_frame.loc[zone1_mask,'speed'])

            lick_mask = sub_frame.lick>0
            pos_lick_mask = lick_mask & (zone0_mask | zone1_mask)
            pos_licks = sub_frame.loc[pos_lick_mask,'pos']
            if pos_licks.shape[0]>0:
                pos_lick[i] = pos_licks.iloc[0]

            if m+wj+bj<.5:
                if rewards[i]>0 and max_pos[i]>rzone1[1]:
                    pcnt[i] = 0
                elif max_pos[i]<rzone1[1]:
                    pcnt[i]=1
            else:
                if rewards[i]>0 and max_pos[i]>rzone1[1]:
                    pcnt[i] = 1
                elif max_pos[i]<rzone1[1]:
                    pcnt[i]=0


            if max_pos[i]>rzone1[1] and rewards[i]==0:
                omissions[i]=1
            ###

    trial_info = {'morphs':morphs,'max_pos':max_pos,'rewards':rewards,'zone0_licks':zone0_licks,'zone1_licks':zone1_licks,'zone0_speed':zone0_speed,
                 'zone1_speed':zone1_speed,'pcnt':pcnt,'wallJitter':wallJitter,'towerJitter':towerJitter,'bckgndJitter':bckgndJitter,'clickOn':clickOn,
                 'pos_lick':pos_lick,'omissions':omissions,'reward_pos':reward_pos,'rzone_entry':rzone_entry}
    return trial_info, tstart_inds, teleport_inds


def avg_by_morph(morphs,mat):
    '''average mat [trials x n ( x m)] by morph values in morphs
    input: morphs - [ntrials,] vector used for binning
            mat - trials x x n (x m) matrix to be binned
    output: pcnt_mean - mat data binned and averaged by values of morphs
    '''

    # account for different sizes of mat
    morphs_u = np.unique(morphs)
    ndim = len(mat.shape)
    if ndim==1:
        pcnt_mean = np.zeros([morphs_u.shape[0],])
    elif ndim==2:
        pcnt_mean = np.zeros([morphs_u.shape[0],mat.shape[1]])
    elif ndim ==3:
        pcnt_mean = np.zeros([morphs_u.shape[0],mat.shape[1],mat.shape[2]])
    else:
        raise(Exception("mat is wrong number of dimensions"))

    #
    for i,m in enumerate(morphs_u):
        if ndim==1:
            pcnt_mean[i] = np.nanmean(mat[morphs==m])
        elif ndim ==2:
            pcnt_mean[i,:] = np.nanmean(mat[morphs==m,:],axis=0)
        elif ndim ==3:
            pcnt_mean[i,:,:]=np.nanmean(mat[morphs==m,:,:],axis=0)
        else:
            pass
    return np.squeeze(pcnt_mean)




def smooth_raster(x,mat,ax=None,smooth=False,sig=2,vals=None,cmap='cool',tports=None):
    '''plot mat ( ntrials x positions) as a smoothed histogram
    inputs: x - positions array (i.e. bin centers)
            mat - trials x positions array to be plotted
            ax - matplotlib axis object to use. if none, create a new figure and new axis
            smooth - bool. smooth raster or not
            sig - width of Gaussian smoothing
            vals - values used to color lines in histogram (e.g. morph value)
            cmap - colormap used appled to vals
            tports - if mouse is teleported between the end of the trial, plot position  of teleport as x
    outpus: ax - axis of plot object'''

    if ax is None:
        f,ax = plt.subplots()

    cm = plt.cm.get_cmap(cmap)

    if smooth:
        k = Gaussian1DKernel(sig)
        for i in range(mat.shape[0]):
            mat[i,:] = convolve(mat[i,:],k,boundary='extend')

    for ind,i in enumerate(np.arange(mat.shape[0]-1,0,-1)):
        if vals is not None:
            ax.fill_between(x,mat[ind,:]+i,y2=i,color=cm(np.float(vals[ind])),linewidth=.001)
        else:
            ax.fill_between(x,mat[ind,:]+i,y2=i,color = 'black',linewidth=.001)

        if tports is not None:
            ax.scatter(tports[ind],i+.5,color=cm(np.float(vals[ind])),marker='x',s=50)

    ax.set_yticks(np.arange(0,mat.shape[0],10))
    ax.set_yticklabels(["%d" % l for l in np.arange(mat.shape[0],0,-10).tolist()])

    return ax


 # def rate_map(C,position,bin_size=5,min_pos = 0, max_pos=450):
 #    '''non-normalized rate map E[df/F]|_x '''
 #    bin_edges = np.arange(min_pos,max_pos+bin_size,bin_size).tolist()
 #    if len(C.shape) ==1:
 #        C = np.expand_dims(C,axis=1)
 #    frmap = np.zeros([len(bin_edges)-1,C.shape[1]])
 #    frmap[:] = np.nan
 #    occupancy = np.zeros([len(bin_edges)-1,])
 #    for i, (edge1,edge2) in enumerate(zip(bin_edges[:-1],bin_edges[1:])):
 #        if np.where((position>edge1) & (position<=edge2))[0].shape[0]>0:
 #            frmap[i] = np.nanmean(C[(position>edge1) & (position<=edge2),:],axis=0)
 #            occupancy[i] = np.where((position>edge1) & (position<=edge2))[0].shape[0]
 #        else:
 #            pass
 #    return frmap, occupancy/occupancy.ravel().sum()




# def morph_pos_rate_map(trial_mat, effMorph):
#     if len(trial_mat.shape)==2:
#         trial_mat=trial_mat[:,:,np.newaxis]
#     effMorph = (effMorph-np.amin(effMorph))/(np.amax(effMorph)-np.amin(effMorph)+.01)+.001
#     morph_edges = np.linspace(.1,1,num=10)
#     ratemap = np.zeros([10,trial_mat.shape[1],trial_mat.shape[2]])
#     ratemap[:]=np.nan
#
#     morph_dig = np.digitize(effMorph,morph_edges)
#     for ind in np.unique(morph_dig).tolist():
#         ratemap[ind,:,:] = np.nanmean(trial_mat[morph_dig==ind,:,:],axis=0)
#     return np.squeeze(ratemap)




# def make_time_bin_trial_matrices(C,tstarts,tstops):
#     if tstarts.shape[0]>1000: # if binary, leaving in for backwards compatibility
#         tstart_inds, tstop_inds = np.where(tstarts==1)[0],np.where(tstops==1)[0]
#         ntrials = np.sum(tstarts)
#     else:
#         tstart_inds, tstop_inds = tstarts, tstops
#         ntrials = tstarts.shape[0]
#     # find longest trial
#     N = (tstops-tstarts).max()
#     if len(C.shape)>1:
#         T = np.zeros([tstarts.shape[0],N,C.shape[1]])
#     else:
#         T = np.zeros([int(tstarts.shape[0]),int(N),1])
#         C = C[:,np.newaxis]
#     T[:]=np.nan
#
#     for t,(start,stop) in enumerate(zip(tstarts.tolist(),tstops.tolist())):
#         l = stop-start
#         T[t,:l,:]=C[start:stop,:]
#     return T
#
#
# def trial_tensor(C,labels,trig_inds,pre=50,post=50):
#     '''create a tensor of trial x time x neural dimension for arbitrary centering indices'''
#
#     if len(C.shape)==1:
#         trialMat = np.zeros([trig_inds.shape[0],pre+post,1])
#         C = np.expand_dims(C,1)
#     else:
#         trialMat = np.zeros([trig_inds.shape[0],pre+post,C.shape[1]])
#     labelVec = np.zeros([trig_inds.shape[0],])
#
#     for ind, t in enumerate(trig_inds):
#         labelVec[ind] = labels[t]
#
#         if t-pre <0:
#             trialMat[ind,pre-t:,:] = C[0:t+post,:]
#             trialMat[ind,0:pre-t,:] = C[0,:]
#
#         elif t+post>C.shape[0]:
#             print(trialMat.shape)
#             print(t, post)
#             print(C.shape[0])
#             print(C[t-pre:,0].shape)
#
#             trialMat[ind,:C.shape[0]-t-post,:] = C[t-pre:,:]
#             trialMat[ind,C.shape[0]-t-post:,:] =  C[-1,:]
#
#         else:
#             trialMat[ind,:,:] = C[t-pre:t+post,:]
#
#     return trialMat, labelVec
#
# def across_trial_avg(trialMat,labelVec):
#     '''use output of trial_tensor function to return trial average'''
#     labels = np.unique(labelVec)
#
#     if len(trialMat.shape)==3:
#         avgMat = np.zeros([labels.shape[0],trialMat.shape[1],trialMat.shape[2]])
#     else:
#         avgMat = np.zeros([labels.shape[0],trialMat.shape[1],1])
#         trialMat = trialMat[:,:,np.newaxis]
#
#     for i, val in enumerate(labels.tolist()):
#         #print(np.where(labelVec==val)[0].shape)
#         avgMat[i,:,:] = np.nanmean(trialMat[labelVec==val,:,:],axis=0)
#
#     return avgMat, labels


#

# def make_spline_basis(x,knots=np.arange(0,1,.2)):
#     '''make cubic spline basis functions'''
#     knotfunc = lambda k: np.power(np.multiply(x-k,(x-k)>0),3)
#     spline_basis_list = [knotfunc(k) for k in knots.tolist()]
#     spline_basis_list += [np.ones(x.shape[0]),x,np.power(x,2)]
#     return np.array(spline_basis_list).T
