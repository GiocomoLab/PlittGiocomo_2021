"""
Non-negative matrix factorization (NMF) analysis of single-cell trial similarity.

Each place cell's trial-by-trial similarity matrix is flattened to a vector
(upper triangle) and stacked across cells and sessions to form a
cells x trial-pairs matrix. Ensemble NMF with cross-validation is used to
find a low-rank decomposition:
  - H factors: basis similarity matrices (one per NMF component)
  - W factors: per-cell loading scores onto each component

This analysis is used to identify stereotyped remapping patterns across the
morphing stimulus axis (e.g. abrupt vs. gradual transitions).
"""

import numpy as np
import matplotlib.pyplot as plt
from . import utilities as u
from . import preprocessing as pp
from . import behavior as b
from . import similarity_matrix_analysis as sm
import scipy as sp
from . import place_cell_analysis as pc
import matplotlib.gridspec as gridspec
import sklearn as sk
import os
from nmftools import ensemble, plots




def run_ensemble(ss_flat, maxfactors=10):
    '''Fit ensemble NMF with cross-validation to determine the optimal rank.

    Runs ensemble.fit_ensemble_cv from nmftools over ranks 1 to maxfactors-1
    with 3 replicates each. Reconstruction error is plotted alongside the SVD
    elbow for reference.

    inputs: ss_flat    - [cells, trial_pairs] non-negative matrix to decompose
            maxfactors - maximum rank to test (exclusive; default 10)
    outputs: results   - dict of NMF results keyed by rank (from fit_ensemble_cv)
             (f, ax)   - figure and axis of the reconstruction error plot
    '''

    results = ensemble.fit_ensemble_cv(ss_flat, np.arange(1,maxfactors),n_replicates=3)
    f,ax = plt.subplots()
    ax = plots.plot_rmse(results,plot_svd=False)

    return results, (f,ax)


def sim_triu(effMorph, S_trial_mat, binned=True, norm=True):
    '''Compute the upper-triangle of a single cell's trial x trial similarity matrix.

    Sorts trials by effective morph value, computes per-position cosine similarity
    between every pair of trials for each cell, then returns the upper-triangle
    entries as a vector. Optionally bins the similarity matrix by morph value
    before extracting the upper triangle.

    inputs: effMorph     - [ntrials,] array of effective morph values (morphs + wallJitter)
            S_trial_mat  - [ntrials, positions, neurons] activity tensor
            binned       - bool; if True, bin the similarity matrix by morph value using
                           u.morph_pos_rate_map before extracting the upper triangle
            norm         - bool; if True, divide each entry by the global mean similarity
    outputs: upper-triangle vector(s), shape depends on binned:
                binned=True  -> [n_unique_morphs*(n_unique_morphs-1)/2, neurons]
                binned=False -> ([triu_pairs, neurons], effMorph sorted)
    '''
    msort = np.argsort(effMorph)
    S_trial_mat=S_trial_mat[msort,:,:]
    S_tm_norm = S_trial_mat/np.linalg.norm(S_trial_mat,ord=2,axis=1)[:,np.newaxis,:]
    S_sim =  np.transpose(np.matmul(np.transpose(S_tm_norm,axes=(2,0,1)),np.transpose(S_tm_norm,axes=(2,1,0))),axes=(1,2,0))
    S_sim[np.isnan(S_sim)]=0
    if norm:
        S_sim/=S_sim.ravel().mean()


    if binned:
        mu_sim = u.morph_pos_rate_map(S_sim,effMorph[msort])
        mu_sim = u.morph_pos_rate_map(np.transpose(mu_sim,axes=(1,0,2)),effMorph[msort])

        ui = np.triu_indices(mu_sim.shape[1],k=1)
        return mu_sim[ui[0],ui[1],:].T
    else:
        ui = np.triu_indices(S_sim.shape[1],k=1)
        return S_sim[ui[0],ui[1],:].T, effMorph[msort]


def build_matrix(df, mouse_list, first_sess=None):
    '''Aggregate upper-triangle similarity vectors across mice and sessions.

    Iterates over a list of mice and their test sessions, calls sim_triu on
    each session's activity tensor, and concatenates the resulting vectors
    along the cells axis to produce a single [total_cells, trial_pairs] matrix
    suitable for NMF decomposition.

    inputs: df         - pandas DataFrame of session metadata (from load_session_db)
            mouse_list - list of mouse identifiers to include (must match df['MouseName'])
            first_sess - starting session index per mouse; int (applied to all mice),
                         list (one per mouse), or None (defaults to 5 for all mice)
    outputs: cellmat   - [total_cells, trial_pairs] concatenated similarity matrix
    '''

    if first_sess is None:
        first_sess = len(mouse_list)*[5]
    elif isinstance(first_sess,int):
        first_sess = len(mouse_list)*[first_sess]
    else:
        pass


    for m, (mouse,_first_sess) in enumerate(zip(mouse_list,first_sess)):
        print(mouse)
        df_mouse = df[df['MouseName'].str.match(mouse)]
        for i, sess_ind in enumerate(range(_first_sess,df_mouse.shape[0])):
            vec = sim_triu(df_mouse.iloc[sess_ind])
            if (m==0) and (i==0):
                cellmat = vec
            else:
                cellmat = np.concatenate((cellmat,vec),axis=0)

    return cellmat


def plot_factors(results, rank, ndim, downsample=1):
    '''Visualize NMF H and W factors for a given rank.

    Reconstructs each H factor (basis similarity matrix) from its upper-triangle
    representation and displays it as a 2-D heatmap. Plots W (per-cell loading
    scores) as scatter plots sorted by each factor's loading.

    inputs: results    - NMF results dict from run_ensemble (keyed by rank)
            rank       - integer rank to visualize (must be a key in results)
            ndim       - number of morph bins (sets the size of reconstructed H matrices)
            downsample - fraction of cells to plot in W scatter plots (default 1 = all)
    outputs: f, ax     - figure and axis array of shape [rank, rank+1]
    '''

    H = np.zeros([rank,ndim,ndim])
    ui = np.triu_indices(ndim,k=1)
    H[:,ui[0],ui[1]]= results[rank]['factors'][0][1]
    H += np.transpose(H,axes=(0,2,1))

    W = results[rank]['factors'][0][0]
    wmax = np.amax(W.ravel())

    Wmask = np.zeros([W.shape[0],])
    rinds = np.random.permutation(W.shape[0])
    Wmask[rinds[:int(W.shape[0]*downsample)]]=1.

    _W = W[Wmask>0,:]


    f,ax = plt.subplots(rank,rank+1,figsize=[5*(rank+1),rank*5])
    for j in range(rank):
        _H = H[j,:,:]
        _H[np.diag_indices_from(_H)]=np.nan
        if rank<2:
            ax[0].imshow(_H,cmap='cividis')
            ax[1].scatter(np.arange(_W.shape[0]),_W.ravel())
        else:
            ax[j,0].imshow(_H,cmap='cividis')
            for k in range(1,rank+1):
                ksort = np.argsort(_W[:,k-1])
                ax[j,k].scatter(np.arange(_W.shape[0]),_W[ksort,j])
                ax[j,k].set_ylim([-.1,wmax+.05])

    return f,ax





def sort_matrix_by_columns(W):
    '''Return row indices that sort W by its last column in ascending order.

    inputs: W - [cells, factors] NMF W matrix
    outputs: sort indices for W rows ordered by W[:, -1]
    '''
    return np.argsort(W[:,-1])
