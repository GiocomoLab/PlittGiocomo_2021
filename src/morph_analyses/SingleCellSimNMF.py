import numpy as np
import matplotlib.pyplot as plt
from . import utilities as u
from . import preprocessing as pp
from . import behavior as b
from . import SimilarityMatrixAnalysis as sm
import scipy as sp
from . import PlaceCellAnalysis as pc
import matplotlib.gridspec as gridspec
import sklearn as sk
import os
# os.sys.path.append("C:\\Users\\markp\\repos\\nmftools")
# from nmftools import ensemble, plots
# import ensemble
# import plots




def run_ensemble(ss_flat,maxfactors = 10):
    '''run ensemble on single session'''

    results = ensemble.fit_ensemble_cv(ss_flat, np.arange(1,maxfactors),n_replicates=3)
    f,ax = plt.subplots()
    ax = plots.plot_rmse(results,plot_svd=False)

    return results, (f,ax)


def sim_triu(effMorph, S_trial_mat, binned = True,norm=True):
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


def build_matrix(df, mouse_list,first_sess=None):

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


def plot_factors(results,rank,ndim,downsample=1):

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

    return np.argsort(W[:,-1])
