import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.ndimage.filters import gaussian_filter1d
import utilities as u
import preprocessing as pp
import PlaceCellAnalysis as pc
import SimilarityMatrixAnalysis as sm
import sklearn as sk
from sklearn.decomposition import PCA
import behavior as b
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle


def set_ops(change_ops={}):
    '''set options for single_session_figs'''
    ops = {'behavior':True,
        'PCA':True,
        'place cells':True,
        'trial simmats':True,
        'savefigs':True,
        'realtime sf':True}
    for k,v in change_ops.items():
        ops[k]=v
    return ops




def single_session_figs(sess,dir = "G:\\My Drive\\Figures\\TwoTower\\SingleSession",
                        dFF=False,ops={},twotower=False):
    '''plot a bunch of useful
    inputs: sess: row from pandas array of session metadata
            can also be a dictionary as long as the following fields are present and valid
            'data file' - raw VR *.sqlite data file path
            'scanmat' - Neurolabware .mat file path
            's2pfolder' - Suite2P output path
            dir: directory for saving figures
            dFF: use deconvolved activity rate (False, default) or fluorescence (True)
            ops: dict, options for which analyses to run and plotting (see set_ops)
            twotower: bool; whether to plot behavior as TwoTower task (True) or foraging (False)
    '''


    # set options
    ops = set_ops(change_ops=ops)
    if ops['savefigs']:
        outdir = os.path.join(dir,sess['MouseName'],"%s_%s_%d" % (sess['Track'],sess['DateFolder'],sess['SessionNumber']))
        try:
            os.makedirs(outdir)
        except:
            print("failed to make path",outdir)


    # load everything up
    VRDat, C, S, A = pp.load_scan_sess(sess,fneu_coeff=.7,analysis='s2p')

    if dFF:
        S = C/1546 # scaling down for better behavior; 1546 = 10*frame rate
    else:
        S/=1546 # scaling down for better behavior; 1546 = 10*frame rate
    S[np.isnan(S)]=0.


    # get trial by trial info
    trial_info, tstart_inds, teleport_inds = u.by_trial_info(VRDat)
    S_trial_mat, occ_trial_mat, edges,centers = u.make_pos_bin_trial_matrices(S,
                                                VRDat['pos']._values,VRDat['tstart']._values,
                                                VRDat['teleport']._values,bin_size=10,
                                                speed = VRDat['speed']._values)
    S_trial_mat[np.isnan(S_trial_mat)]=0
    S_morph_dict = u.trial_type_dict(S_trial_mat,trial_info['morphs'])
    occ_morph_dict = u.trial_type_dict(occ_trial_mat,trial_info['morphs'])

    effMorph = trial_info['morphs']+trial_info['wallJitter']+trial_info['bckgndJitter']+trial_info['towerJitter']
    reward_pos = trial_info['rzone_entry']
    reward_pos[np.isnan(reward_pos)]= 480

    if ops['behavior']:
        #lick data
        lick_trial_mat= u.make_pos_bin_trial_matrices(VRDat['lick']._values,
                                                            VRDat['pos']._values,
                                                            VRDat['tstart']._values,
                                                            VRDat['teleport']._values,
                                                            mat_only=True,bin_size=10)
        lick_morph_dict = u.trial_type_dict(lick_trial_mat,trial_info['morphs'])
        max_pos = np.copy(trial_info['max_pos'])
        max_pos[max_pos>440]=np.nan

        # plot speed data
        speed_trial_mat = u.make_pos_bin_trial_matrices(VRDat['speed']._values,
                                                        VRDat['pos']._values,
                                                        VRDat['tstart']._values,
                                                        VRDat['teleport']._values,
                                                        mat_only=True,bin_size=10)
        speed_morph_dict = u.trial_type_dict(speed_trial_mat,trial_info['morphs'])


        # plot lick and speed rasters depending on which task was run
        if twotower:
            f_lick, axarr_lick = b.behavior_raster_task(lick_trial_mat/np.nanmax(lick_trial_mat.ravel()),
                                                    centers,effMorph,reward_pos/390.,smooth=False)

            f_speed,axarr_speed = b.behavior_raster_task(speed_trial_mat/np.nanmax(speed_trial_mat.ravel()),
                                                    centers,effMorph,reward_pos/390.,smooth=False)

        else:
            f_lick, axarr_lick = b.behavior_raster_foraging(lick_trial_mat/np.nanmax(lick_trial_mat.ravel()),
                                                    centers,effMorph,reward_pos/390.,smooth=False)

            f_speed,axarr_speed = b.behavior_raster_foraging(speed_trial_mat/np.nanmax(speed_trial_mat.ravel()),
                                                    centers,effMorph,reward_pos/390.,smooth=False)
        if ops['savefigs']:
            f_lick.savefig(os.path.join(outdir,'licks.pdf'),format='pdf')
            f_speed.savefig(os.path.join(outdir,'speed.pdf'),format='pdf')

    if ops['PCA']:
        # PCA
        f_pca,[ax_pca, aax_pca, aaax_pca] = plot_pca(C,VRDat)
        if ops['savefigs']:
            f_pca.savefig(os.path.join(outdir,'PCA.pdf'),format='pdf')

    if ops['place cells']:
        # find place cells
        masks, FR, SI = pc.place_cells_calc(S, VRDat['pos']._values,trial_info,
                        VRDat['tstart']._values, VRDat['teleport']._values,
                        method='bootstrap',correct_only=False,speed=VRDat.speed._values,
                        win_trial_perm=True,morphlist=np.unique(trial_info['morphs']).tolist())

        # plot place cells by morph
        f_pc, ax_pc,PC_dict = pc.plot_placecells(S_morph_dict,masks)

        # number in each environment
        print('morph 0 place cells = %g out of %g , %f ' % (masks[0].sum(), masks[0].shape[0], masks[0].sum()/masks[0].shape[0]))
        print('morph 1 place cells = %g out of %g, %f' % (masks[1].sum(), masks[1].shape[0], masks[1].sum()/masks[1].shape[0]))

        # single cell plots
        f_singlecells = pc.plot_top_cells(S_trial_mat,masks,SI,effMorph)

        # position by morph similarity matrix averaging trials
        SM = sm.morph_simmat(S_morph_dict,corr=False)
        m=np.unique(trial_info['morphs']).size
        U  = sm.morph_mean_simmat(SM,m)
        f_SM, ax_SM = sm.plot_simmat(SM,m)
        f_U,ax_U = plt.subplots()
        ax_U.imshow(U,cmap='pink')

        if ops['savefigs']:
            f_pc.savefig(os.path.join(outdir,'placecells.pdf'),format='pdf')
            f_singlecells.savefig(os.path.join(outdir,'singlecells.pdf'),format='pdf')
            f_SM.savefig(os.path.join(outdir,'morphxpos_simmat.pdf'),format='pdf')
            f_U.savefig(os.path.join(outdir,'morph_simmat.pdf'),format='pdf')
            with open(os.path.join(outdir,'pc_masks.pkl'),'wb') as f:
                pickle.dump({'masks':masks},f)

    if ops['trial simmats']:
        # trial by trial similarity matrix
        rmask = trial_info['rewards']==0
        S_trial_mat[np.isnan(S_trial_mat)]=0
        # S_trial_mat is trials x positions x neurons
        S_trial_mat = sp.ndimage.filters.gaussian_filter1d(S_trial_mat,1,axis=1) # smooth position by 1 bin

        #flatten to be trial x positions*neurons
        S_tmat = np.reshape(S_trial_mat,[S_trial_mat.shape[0],-1])
        # divide trials by l2-norm
        S_tmat = S_tmat/np.linalg.norm(S_tmat,ord=2,axis=-1)[:,np.newaxis]
        # outer product give trial x trial cosine similarity
        S_t_rmat = np.matmul(S_tmat,S_tmat.T)
        # ensure that this is positive (should be true if using deconvolved rate)
        print("negative similarity inds:", (S_t_rmat<0).ravel().sum())

        # plot results
        f_stsm,axtup_stsm = sm.plot_trial_simmat(S_t_rmat,trial_info)

        # plot similarity fraction
        sf = np.zeros(effMorph.shape) # similarity fraction
        for trial in range(effMorph.shape[0]): # for each trial
            # get masks for centroids
            mask0 = trial_info['morphs']==0
            mask1 = trial_info['morphs']==1
            # if current trial is in mask, exclude it
            if trial_info['morphs'][trial]==0:
                mask0[trial]=False
            elif trial_info['morphs'][trial]==1:
                mask1[trial]=False

            # calculate centroids
            centroid0, centroid1 = np.nanmean(S_tmat[mask0,:],axis=0), np.nanmean(S_tmat[mask1,:],axis=0)
            centroid0/np.linalg.norm(centroid0,ord=2)
            centroid1/np.linalg.norm(centroid1,ord=2)

            # similarity to two centroids
            angle0,angle1 = np.dot(S_tmat[trial,:],centroid0),np.dot(S_tmat[trial,:],centroid1)
            # whole trial similarity fraction
            sf[trial] = angle0/(angle0+angle1)

        f_sf,ax_sf = plt.subplots()
        ax_sf.scatter(effMorph,sf,c=1-effMorph,cmap='cool')
        ax_sf.scatter(effMorph[rmask],sf[rmask],c='black')

        # eigendecomposition of trial x trial cosine similarity matrix
        [w,V]=np.linalg.eig(S_t_rmat)
        order = np.argsort(w)[::-1]
        w = w[order]
        V=V[:,order]
        # project trials onto top 3 eigenvectors
        X =  np.matmul(S_t_rmat,V[:,:3])
        f_embed = plt.figure(figsize=[20,20])
        ax_embed3d = f_embed.add_subplot(221, projection='3d')
        ax_embed3d.scatter(X[:,0],X[:,1],X[:,2],c=1-effMorph,cmap='cool')


        ax_embed3d.scatter(X[rmask,0],X[rmask,1],X[rmask,2],c='black')
        ax_embed2d = f_embed.add_subplot(222)
        ax_embed2d.scatter(X[:,0],X[:,1],c=1-effMorph,cmap='cool')
        ax_embed2d.scatter(X[rmask,0],X[rmask,1],c='black')
        ax_embed3d = f_embed.add_subplot(223, projection='3d')
        ax_embed3d.scatter(X[:,0],X[:,1],X[:,2],c=np.arange(X.shape[0]),cmap='viridis')
        ax_embed2d = f_embed.add_subplot(224)
        ax_embed2d.scatter(X[:,0],X[:,1],c=np.arange(X.shape[0]),cmap='viridis')

        # try a nonlinear embedding too
        lem = sk.manifold.SpectralEmbedding(affinity='precomputed',n_components=3)
        X = lem.fit_transform(S_t_rmat)

        f_se = plt.figure(figsize=[20,20])
        ax_se3d = f_se.add_subplot(221, projection='3d')
        ax_se3d.scatter(X[:,0],X[:,1],X[:,2],c=1-effMorph,cmap='cool')
        ax_se3d.scatter(X[rmask,0],X[rmask,1],X[rmask,2],c='black')

        ax_se2d = f_se.add_subplot(222)
        ax_se2d.scatter(X[:,0],X[:,1],c=1-effMorph,cmap='cool')
        ax_se2d.scatter(X[rmask,0],X[rmask,1],c='black')

        ax_se3d = f_se.add_subplot(223, projection='3d')
        ax_se3d.scatter(X[:,0],X[:,1],X[:,2],c=np.arange(X.shape[0]),cmap='viridis')
        ax_se2d = f_se.add_subplot(224)
        ax_se2d.scatter(X[:,0],X[:,1],c=np.arange(X.shape[0]),cmap='viridis')

        if ops['savefigs']:
            f_stsm.savefig(os.path.join(outdir,'trial_simmat.pdf'),format='pdf')
            f_embed.savefig(os.path.join(outdir,'simmat_embed.pdf'),format='pdf')
            f_se.savefig(os.path.join(outdir,'simmat_spectembed.pdf'),format='pdf')
            f_sf.savefig(os.path.join(outdir,'sf.pdf'),format='pdf')

    if ops['realtime sf']:
        # plot similarity fraction as a function of position
        if ops['trial simmats']:
            pass
        else:
            S_trial_mat = sp.ndimage.filters.gaussian_filter1d(S_trial_mat,1,axis=1)


        sf_bin = np.zeros(S_trial_mat.shape[:-1])
        # divide population by l2 norm at each position bin
        S_tmat_norm = S_trial_mat/np.linalg.norm(S_trial_mat,2,axis=-1)[:,:,np.newaxis]
        for trial in range(S_trial_mat.shape[0]):
            # cross-validate masks for centroids
            mask0 = trial_info['morphs']==0
            mask1 = trial_info['morphs']==1
            if trial_info['morphs'][trial]==0:
                mask0[trial]=False
            elif trial_info['morphs'][trial]==1:
                mask1[trial]=False


            # calculate centroids
            centroid_0 = S_trial_mat[mask0,:,:].mean(axis=0)
            centroid_0/=np.linalg.norm(centroid_0,2,axis=-1)[:,np.newaxis]

            centroid_1 = S_trial_mat[mask1,:,:].mean(axis=0)
            centroid_1/=np.linalg.norm(centroid_1,2,axis=-1)[:,np.newaxis]

            # calculate similarity fraction
            angle0 = np.diagonal(np.matmul(S_tmat_norm[trial,:,:],centroid_0.T))
            angle1 = np.diagonal(np.matmul(S_tmat_norm[trial,:,:],centroid_1.T))
            sf_bin[trial,:]=angle0/(angle1+angle0)


        f_rtsf,ax_rtsf = plt.subplots()
        for t in range(sf_bin.shape[0]):
            ax_rtlar.plot(sf_bin[t,:],c=plt.cm.cool(1-effMorph[t]),alpha=.3)

        if ops['savefigs']:
            f_rtsf.savefig(os.path.join(outdir,'rt_sf.pdf'),format='pdf')


    if ops['trial simmats'] and ops['place cells']:

        # plot trial x trial similarity results using only place cells
        cellmask = np.zeros([S.shape[1],])>1
        for k,v in masks.items(): # masks is from pc.place_cells_calc
            cellmask = cellmask | v

        S_trial_mat_pc = S_trial_mat[:,:,cellmask]
        # flatten, normalize, outer product  as above
        #        exclude possible end of track problems for scenes where there was a timeout
        if sess['Track'] in ('TwoTower_noTimeout','TwoTower_Timeout','FreqMorph_Timeout','FreqMorph_Decision'):
            S_tmat = np.reshape(S_trial_mat_pc[:,:20,:],[S_trial_mat.shape[0],-1])
        else:
            S_tmat = np.reshape(S_trial_mat_pc,[S_trial_mat.shape[0],-1])
        S_tmat = S_tmat/np.linalg.norm(S_tmat,ord=2,axis=-1)[:,np.newaxis]
        S_t_rmat = np.matmul(S_tmat,S_tmat.T)

        f_stsm,axtup_stsm = sm.plot_trial_simmat(S_t_rmat,trial_info)

        sf = np.zeros(effMorph.shape)
        for trial in range(effMorph.shape[0]):
            mask0 = trial_info['morphs']==0
            mask1 = trial_info['morphs']==1
            if trial_info['morphs'][trial]==0:
                mask0[trial]=False
            elif trial_info['morphs'][trial]==1:
                mask1[trial]=False

            # calculate centroids
            centroid0, centroid1 = np.nanmean(S_tmat[mask0,:],axis=0), np.nanmean(S_tmat[mask1,:],axis=0)
            centroid0/np.linalg.norm(centroid0,ord=2)
            centroid1/np.linalg.norm(centroid1,ord=2)

            # similarity to two centroids
            angle0,angle1 = np.dot(S_tmat[trial,:],centroid0),np.dot(S_tmat[trial,:],centroid1)
            # whole trial similarity fraction
            sf[trial] = angle0/(angle0+angle1)

        f_sf,ax_sf = plt.subplots()
        ax_sf.scatter(effMorph,sf,c=1-effMorph,cmap='cool')
        ax_sf.scatter(effMorph[rmask],sf[rmask],c='black')

        [w,V]=np.linalg.eig(S_t_rmat)
        order = np.argsort(w)[::-1]
        w = w[order]
        V=V[:,order]
        X =  np.matmul(S_t_rmat,V[:,:3])
        f_embed = plt.figure(figsize=[20,20])
        ax_embed3d = f_embed.add_subplot(221, projection='3d')
        ax_embed3d.scatter(X[:,0],X[:,1],X[:,2],c=1-effMorph,cmap='cool')


        ax_embed3d.scatter(X[rmask,0],X[rmask,1],X[rmask,2],c='black')
        ax_embed2d = f_embed.add_subplot(222)
        ax_embed2d.scatter(X[:,0],X[:,1],c=1-effMorph,cmap='cool')
        ax_embed2d.scatter(X[rmask,0],X[rmask,1],c='black')
        ax_embed3d = f_embed.add_subplot(223, projection='3d')
        ax_embed3d.scatter(X[:,0],X[:,1],X[:,2],c=np.arange(X.shape[0]),cmap='viridis')
        ax_embed2d = f_embed.add_subplot(224)
        ax_embed2d.scatter(X[:,0],X[:,1],c=np.arange(X.shape[0]),cmap='viridis')

        lem = sk.manifold.SpectralEmbedding(affinity='precomputed',n_components=3)
        X = lem.fit_transform(S_t_rmat)

        f_se = plt.figure(figsize=[20,20])
        ax_se3d = f_se.add_subplot(221, projection='3d')
        ax_se3d.scatter(X[:,0],X[:,1],X[:,2],c=1-effMorph,cmap='cool')
        ax_se3d.scatter(X[rmask,0],X[rmask,1],X[rmask,2],c='black')

        ax_se2d = f_se.add_subplot(222)
        ax_se2d.scatter(X[:,0],X[:,1],c=1-effMorph,cmap='cool')
        ax_se2d.scatter(X[rmask,0],X[rmask,1],c='black')

        ax_se3d = f_se.add_subplot(223, projection='3d')
        ax_se3d.scatter(X[:,0],X[:,1],X[:,2],c=np.arange(X.shape[0]),cmap='viridis')
        ax_se2d = f_se.add_subplot(224)
        ax_se2d.scatter(X[:,0],X[:,1],c=np.arange(X.shape[0]),cmap='viridis')

        if ops['savefigs']:
            f_stsm.savefig(os.path.join(outdir,'trial_simmat_pc.pdf'),format='pdf')
            f_embed.savefig(os.path.join(outdir,'simmat_embed_pc.pdf'),format='pdf')
            f_se.savefig(os.path.join(outdir,'simmat_spectembed_pc.pdf'),format='pdf')
            f_sf.savefig(os.path.join(outdir,'sf_pc.pdf'),format='pdf')

    return



def plot_pca(C,VRDat):
    pca = PCA()
    trialMask = (VRDat['pos']>0) & (VRDat['pos']<445)
    print(np.isnan(C).sum(),np.isinf(C).sum())
    X = pca.fit_transform(C/np.amax(C))

    print(X.shape)
    f = plt.figure()
    axarr = []

    XX = X[trialMask[:X.shape[0]]]
    XX = XX[::5,:]
    morph = VRDat.loc[trialMask,'morph']._values
    morph = morph[::5]
    pos = VRDat.loc[trialMask,'pos']._values
    pos = pos[::5]


    time = VRDat.loc[trialMask,'time']._values
    time = time[::5]


    ax=f.add_subplot(131,projection='3d')
    s_cxt=ax.scatter(XX[:,0],XX[:,1],XX[:,2],c=morph,cmap='cool',s=2)

    aax = f.add_subplot(132,projection='3d')
    s_pos=aax.scatter(XX[:,0],XX[:,1],XX[:,2],c=pos,cmap='magma',s=2)

    aaax = f.add_subplot(133,projection='3d')
    s_pos=aaax.scatter(XX[:,0],XX[:,1],XX[:,2],c=time,cmap='viridis',s=2)

    return f,[ax, aax, aaax]
