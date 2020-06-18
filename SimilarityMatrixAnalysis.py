import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.ndimage.filters import gaussian_filter
import sklearn as sk
import sklearn.cluster as clust
import matplotlib.gridspec as gridspec



def plot_trial_simmat(C,trial_info,vmax=None,vmin=None):
    '''Plot trial by trial similarity matrix, C. Produces three plots
    1) trials sorted by the order in which they occured
    2) trials sorted morph value
    3) trials are clustered by spectral clustering (assumes C is non-negative).
        Uses silhouette_score as a heuristic for choosing the number of clusters
    inputs: C - trial by trial similarity matrix (typically cosine similarity)
            trial_info - output of utilities.by_trial_info
            vmax - max for colormap
            vmin - min for colormap
    outputs: f, axlist - figure and axis objects
    '''

    if vmax is None:
        vmax = np.percentile(C.ravel(),90)
    if vmin is None:
        vmin = np.percentile(C.ravel(),10)


    f = plt.figure(figsize=[30,12])
    gs = gridspec.GridSpec(14,30)

    effMorph = trial_info['morphs'] + trial_info['wallJitter'] #+ trial_info['bckgndJitter'] + trial_info['towerJitter']
    effMorph = (effMorph+.1)/1.2
    msort = np.argsort(effMorph)

    x=np.arange(effMorph.size)
    rmask = trial_info['rewards']>0
    tnumber = np.arange(x.shape[0])/x.shape[0]


    ##### sort by trial order
    c_ax = f.add_subplot(gs[:10,:10])
    c_ax.imshow(C,cmap='Greys',vmin=vmin,vmax=vmax,aspect='auto')
    c_ax.set_yticks([])
    c_ax.set_xticks([])

    m_ax = f.add_subplot(gs[10:12,:10])
    m_ax.scatter(x,effMorph,c=1-effMorph,cmap='cool')
    m_ax.scatter(x[~rmask],effMorph[~rmask],c='black',s=10)
    m_ax.set_xlim([0,x.shape[0]])
    m_ax.set_yticks([])
    m_ax.set_xticks([])

    # put black dots on unrewarded trials
    t_ax = f.add_subplot(gs[12:,:10])
    t_ax.scatter(x,tnumber,c=tnumber,cmap='viridis')
    t_ax.scatter(x[~rmask],tnumber[~rmask],c='black',s=10)
    t_ax.set_xlim([0,x.shape[0]])
    t_ax.set_yticks([])

    # sort similarity matrix by morph
    C_msort = _sort_simmat(C,msort)

    ###### sort by morph value
    cm_ax = f.add_subplot(gs[:10,10:20])
    cm_ax.imshow(C_msort,cmap='Greys',vmin=vmin,vmax=vmax,aspect='auto')
    cm_ax.set_yticks([])
    cm_ax.set_xticks([])

    mm_ax = f.add_subplot(gs[10:12,10:20])
    mm_ax.scatter(x,effMorph[msort],c=1-effMorph[msort],cmap='cool')
    emr = np.copy(effMorph)
    emr[rmask]=np.nan

    mm_ax.scatter(x,emr[msort],c='black',s=10)
    mm_ax.set_xlim([0,x.shape[0]])
    mm_ax.set_yticks([])
    mm_ax.set_xticks([])
    tm_ax = f.add_subplot(gs[12:,10:20])
    tm_ax.scatter(x,tnumber[msort],c=tnumber[msort],cmap='viridis')
    tm_ax.set_xlim([0,x.shape[0]])
    tm_ax.set_yticks([])

    # sort similarity matrix by cluster - laplacian eigenmaps
    
    clustsort = _sort_clusters(cluster_simmat(C),effMorph)
    C_csort = _sort_simmat(C,clustsort)

    ##### sort by morph value of cluster
    cc_ax = f.add_subplot(gs[:10,20:])
    cc_ax.imshow(C_csort,cmap='Greys',vmin=vmin,vmax=vmax,aspect='auto')
    cc_ax.set_yticks([])
    cc_ax.set_xticks([])
    mc_ax = f.add_subplot(gs[10:12,20:])
    mc_ax.scatter(x,effMorph[clustsort],c=1-effMorph[clustsort],cmap='cool')
    mc_ax.scatter(x,emr[clustsort],c='black',s=10)
    mc_ax.set_yticks([])
    mc_ax.set_xticks([])
    mc_ax.set_xlim([0,x.shape[0]])
    tc_ax = f.add_subplot(gs[12:,20:])
    tc_ax.scatter(x,tnumber[clustsort],c=tnumber[clustsort],cmap='viridis')
    tc_ax.set_xlim([0,x.shape[0]])
    tc_ax.set_yticks([])

    return f, [[c_ax,m_ax,t_ax],[cm_ax,mm_ax,tm_ax],[cc_ax,mc_ax,tc_ax]]

def cluster_simmat(C):
    '''
    Perform spectral clustering on matrix C. Output labels from clustering.
    Assumes C is elementwise non-negative
    '''
    score = []
    for c in range(2,10): # loop through possible numbers of clusters

        spectclust = clust.SpectralClustering(n_clusters=c,affinity='precomputed')
        labels = spectclust.fit_predict(C)
        s=sk.metrics.silhouette_score(1-C,labels,metric='precomputed')
        score.append(np.floor(100.*s)) # round silhouette_score
        print(s*100.)

    # choose number of clusters
    c = np.argmax(score)+2
    spectclust = clust.SpectralClustering(n_clusters=c,affinity='precomputed')
    spectclust.fit(C)
    return spectclust.labels_

def _sort_clusters(clustlabels,metric):
    '''
    sort clusters in clustlabels by value of metric
    '''
    nc = np.unique(clustlabels).shape[0]
    clustmean = np.array([metric[clustlabels==i].mean() for i in range(nc)])
    clusterOrder = np.argsort(clustmean)
    labels = np.zeros(metric.shape)

    for i,cl in enumerate(clusterOrder.tolist()):
        labels[clustlabels==cl]=i

    return np.argsort(labels)


def _sort_simmat(A,sort):
    ''' sort rows then columns of A by sort'''
    A = A[sort,:]
    return A[:,sort]


def trial_simmat(S_tm):
    '''calculate trial by trial cosine similarity matrix for trials x (positions x neurons) matrix/tensor'''
    # flatten across cells
    S_mat = S_tm.reshape([S_tm.shape[0],-1])
    # normalize by L2 norm
    S_mat/=np.linalg.norm(S_mat,ord=2,axis=1)[:,np.newaxis]
    return np.matmul(S_mat,S_mat.T)
