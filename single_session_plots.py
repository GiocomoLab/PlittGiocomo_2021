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
        'realtime lar':True}
    for k,v in change_ops.items():
        ops[k]=v
    return ops




def single_session_figs(sess,dir = "G:\\My Drive\\Figures\\TwoTower\\SingleSession", ops={},twotower=False):
    '''plot a bunch of useful
    inputs: sess: row from pandas array of session metadata
            can also be a dictionary as long as the following fields are present and valid
            'data file' - raw VR *.sqlite data file path
            'scanmat' - Neurolabware .mat file path
            's2pfolder' - Suite2P output path
            dir: directory for saving figures
            ops: dict, options for which analyses to run and plotting (see set_ops)
            twotower: bool; whether to plot behavior as TwoTower task (True) or foraging (False)
    '''
    ops = set_ops(change_ops=ops)
    if ops['savefigs']:
        outdir = os.path.join(dir,sess['MouseName'],"%s_%s_%d" % (sess['Track'],sess['DateFolder'],sess['SessionNumber']))

        try:
            os.makedirs(outdir)
        except:
            print("failed to make path",outdir)


    # load everything up
    VRDat, C, S, A = pp.load_scan_sess(sess,fneu_coeff=.7,analysis='s2p')

    S/=1546
    # S = S/np.percentile(S,95,axis=0)[np.newaxis,:]
    S[np.isnan(S)]=0.
    C[np.isnan(C)]=0.
    # S=C
    # C/=1546
    print("neg inds:", (A<0).ravel().sum())
    # S /= np.nanmean(S,axis=0)[np.newaxis,:]
    # get trial by trial info
    trial_info, tstart_inds, teleport_inds = u.by_trial_info(VRDat)
    S_trial_mat, occ_trial_mat, edges,centers = u.make_pos_bin_trial_matrices(S,
                                                VRDat['pos']._values,VRDat['tstart']._values,
                                                VRDat['teleport']._values,bin_size=10,
                                                speed = VRDat['speed']._values)
    S_trial_mat[np.isnan(S_trial_mat)]=0
    S_morph_dict = u.trial_type_dict(S_trial_mat,trial_info['morphs'])
    occ_morph_dict = u.trial_type_dict(occ_trial_mat,trial_info['morphs'])

    effMorph = trial_info['morphs']+trial_info['wallJitter']+trial_info['bckgndJitter']+trial_info['towerJitter']#+.3)/1.6
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
        pcnt = u.correct_trial_mask(trial_info['rewards'],tstart_inds,teleport_inds,S.shape[0])
        f_pca,[ax_pca, aax_pca, aaax_pca] = plot_pca(C,VRDat,np.array([]),plot_err=False)

        if ops['savefigs']:
            f_pca.savefig(os.path.join(outdir,'PCA.pdf'),format='pdf')

    if ops['place cells']:
        ################ Place cells
        masks, FR, SI = pc.place_cells_calc(S, VRDat['pos']._values,trial_info,
                        VRDat['tstart']._values, VRDat['teleport']._values,
                        method='bootstrap',correct_only=False,speed=VRDat.speed._values,
                        win_trial_perm=True,morphlist=np.unique(trial_info['morphs']).tolist())

        # plot place cells by morph
        f_pc, ax_pc,PC_dict = pc.plot_placecells(S_morph_dict,masks)

        # number in each environment
        print('morph 0 place cells = %g out of %g , %f ' % (masks[0].sum(), masks[0].shape[0], masks[0].sum()/masks[0].shape[0]))
        print('morph 1 place cells = %g out of %g, %f' % (masks[1].sum(), masks[1].shape[0], masks[1].sum()/masks[1].shape[0]))

        # reward cell plot
        # make tensor for reward location centered position



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
        S_trial_mat = sp.ndimage.filters.gaussian_filter1d(S_trial_mat,1,axis=1)
        # if sess['Track'] in ('TwoTower_Timeout','FreqMorph_Timeout'):
        #     S_tmat = np.reshape(S_trial_mat[:,:20,:],[S_trial_mat.shape[0],-1])
        # else:
        S_tmat = np.reshape(S_trial_mat,[S_trial_mat.shape[0],-1])
        S_tmat = S_tmat/np.linalg.norm(S_tmat,ord=2,axis=-1)[:,np.newaxis]
        S_t_rmat = np.matmul(S_tmat,S_tmat.T)
        print("negative similarity inds:", (S_t_rmat<0).ravel().sum())

        f_stsm,axtup_stsm = sm.plot_trial_simmat(S_t_rmat,trial_info)

        lar = np.zeros(effMorph.shape)
        for trial in range(effMorph.shape[0]):
            mask0 = trial_info['morphs']==0
            mask1 = trial_info['morphs']==1
            if trial_info['morphs'][trial]==0:
                mask0[trial]=False
            elif trial_info['morphs'][trial]==1:
                mask1[trial]=False

            centroid0, centroid1 = np.nanmean(S_tmat[mask0,:],axis=0), np.nanmean(S_tmat[mask1,:],axis=0)
            centroid0/np.linalg.norm(centroid0,ord=2)
            centroid1/np.linalg.norm(centroid1,ord=2)

            angle0,angle1 = np.dot(S_tmat[trial,:],centroid0),np.dot(S_tmat[trial,:],centroid1)
            lar[trial] = angle0/(angle0+angle1)
            # lar[trial]= np.log(np.dot(S_tmat[trial,:],centroid0)/np.dot(S_tmat[trial,:],centroid1))

        f_lar,ax_lar = plt.subplots()
        ax_lar.scatter(effMorph,lar,c=1-effMorph,cmap='cool')
        ax_lar.scatter(effMorph[rmask],lar[rmask],c='black')
        # spectral embedding of single trial similarity matrix
        # lem = sk.manifold.SpectralEmbedding(affinity='precomputed',n_components=3)
        # X = lem.fit_transform(S_t_rmat)

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
            f_stsm.savefig(os.path.join(outdir,'trial_simmat.pdf'),format='pdf')
            f_embed.savefig(os.path.join(outdir,'simmat_embed.pdf'),format='pdf')
            f_se.savefig(os.path.join(outdir,'simmat_spectembed.pdf'),format='pdf')
            f_lar.savefig(os.path.join(outdir,'lar.pdf'),format='pdf')

    if ops['realtime lar']:
        if ops['trial simmats']:
            pass
        else:
            S_trial_mat = sp.ndimage.filters.gaussian_filter1d(S_trial_mat,1,axis=1)

        lr_bin = np.zeros(S_trial_mat.shape[:-1])
        S_tmat_norm = S_trial_mat/np.linalg.norm(S_trial_mat,2,axis=-1)[:,:,np.newaxis]
        for trial in range(S_trial_mat.shape[0]):
            mask0 = trial_info['morphs']==0
            mask1 = trial_info['morphs']==1
            if trial_info['morphs'][trial]==0:
                mask0[trial]=False
            elif trial_info['morphs'][trial]==1:
                mask1[trial]=False


            centroid_0 = S_trial_mat[mask0,:,:].mean(axis=0)
            centroid_0/=np.linalg.norm(centroid_0,2,axis=-1)[:,np.newaxis]

            centroid_1 = S_trial_mat[mask1,:,:].mean(axis=0)
            centroid_1/=np.linalg.norm(centroid_1,2,axis=-1)[:,np.newaxis]

            angle0 = np.diagonal(np.matmul(S_tmat_norm[trial,:,:],centroid_0.T))
            angle1 = np.diagonal(np.matmul(S_tmat_norm[trial,:,:],centroid_1.T))
            lr_bin[trial,:]=angle0/(angle1+angle0)
            # lr_bin[trial,:]=np.log(angle1/angle0)

        f_rtlar,ax_rtlar = plt.subplots()
        for t in range(lr_bin.shape[0]):
            ax_rtlar.plot(lr_bin[t,:],c=plt.cm.cool(1-effMorph[t]),alpha=.3)

        if ops['savefigs']:
            f_rtlar.savefig(os.path.join(outdir,'rt_lar.pdf'),format='pdf')


    if ops['trial simmats'] and ops['place cells']:
        # trial by trial similarity matrix
        cellmask = np.zeros([S.shape[1],])>1
        for k,v in masks.items():
            cellmask = cellmask | v

        S_trial_mat_pc = S_trial_mat[:,:,cellmask]
        if sess['Track'] in ('TwoTower_noTimeout','TwoTower_Timeout','FreqMorph_Timeout','FreqMorph_Decision'):
            S_tmat = np.reshape(S_trial_mat_pc[:,:20,:],[S_trial_mat.shape[0],-1])
        else:
            S_tmat = np.reshape(S_trial_mat_pc,[S_trial_mat.shape[0],-1])
        S_tmat = S_tmat/np.linalg.norm(S_tmat,ord=2,axis=-1)[:,np.newaxis]
        S_t_rmat = np.matmul(S_tmat,S_tmat.T)

        f_stsm,axtup_stsm = sm.plot_trial_simmat(S_t_rmat,trial_info)

        lar = np.zeros(effMorph.shape)
        for trial in range(effMorph.shape[0]):
            mask0 = trial_info['morphs']==0
            mask1 = trial_info['morphs']==1
            if trial_info['morphs'][trial]==0:
                mask0[trial]=False
            elif trial_info['morphs'][trial]==1:
                mask1[trial]=False

            centroid0, centroid1 = np.nanmean(S_tmat[mask0,:],axis=0), np.nanmean(S_tmat[mask1,:],axis=0)
            centroid0/=np.linalg.norm(centroid0,ord=2)
            centroid1/=np.linalg.norm(centroid1,ord=2)

            lar[trial]= np.log(np.dot(S_tmat[trial,:],centroid0)/np.dot(S_tmat[trial,:],centroid1))

        f_lar,ax_lar = plt.subplots()
        ax_lar.scatter(effMorph,lar,c=1-effMorph,cmap='cool')
        ax_lar.scatter(effMorph[rmask],lar[rmask],c='black')
        # spectral embedding of single trial similarity matrix
        # lem = sk.manifold.SpectralEmbedding(affinity='precomputed',n_components=3)
        # X = lem.fit_transform(S_t_rmat)

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
            f_lar.savefig(os.path.join(outdir,'lar_pc.pdf'),format='pdf')

    return



def plot_pca(C,VRDat,pcnt,plot_err=False):
    pca = PCA()
    trialMask = (VRDat['pos']>0) & (VRDat['pos']<445)
    print(np.isnan(C).sum(),np.isinf(C).sum())
    X = pca.fit_transform(C/np.amax(C))

    print(X.shape)
    # skree plots
    f = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #f,axarr = plt.subplots(2,2,figsize=[15,15])
    axarr = []

    XX = X[trialMask[:X.shape[0]]]
    XX = XX[::5,:]
    morph = VRDat.loc[trialMask,'morph']._values
    morph = morph[::5]
    pos = VRDat.loc[trialMask,'pos']._values
    pos = pos[::5]


    time = VRDat.loc[trialMask,'time']._values
    time = time[::5]

    if plot_err:
        print(pcnt.shape)
        pcnt = pcnt[trialMask]
        pcnt = pcnt[::5]
        print(pcnt.shape,XX.shape)
        ax=f.add_subplot(141,projection='3d')
        s_cxt=ax.scatter(XX[pcnt>0,0],XX[pcnt>0,1],XX[pcnt>0,2],c=morph[pcnt>0],cmap='cool',s=2,alpha=1)
        ax_e = f.add_subplot(142,projection='3d')
        s_cxt=ax_e.scatter(XX[pcnt>0,0],XX[pcnt>0,1],XX[pcnt>0,2],c=morph[pcnt>0],cmap='cool',s=2,alpha=.01)
        s_cxt=ax_e.scatter(XX[pcnt<1,0],XX[pcnt<1,1],XX[pcnt<1,2],c=morph[pcnt<1],cmap='cool',s=2,alpha=1)

        aax = f.add_subplot(143,projection='3d')
        s_pos=aax.scatter(XX[:,0],XX[:,1],XX[:,2],c=pos,cmap='magma',s=2)

        aaax = f.add_subplot(144,projection='3d')
        s_pos=aaax.scatter(XX[:,0],XX[:,1],XX[:,2],c=time,cmap='viridis',s=2)

    else:
        ax=f.add_subplot(131,projection='3d')
        s_cxt=ax.scatter(XX[:,0],XX[:,1],XX[:,2],c=morph,cmap='cool',s=2)


        aax = f.add_subplot(132,projection='3d')
        s_pos=aax.scatter(XX[:,0],XX[:,1],XX[:,2],c=pos,cmap='magma',s=2)

        aaax = f.add_subplot(133,projection='3d')
        s_pos=aaax.scatter(XX[:,0],XX[:,1],XX[:,2],c=time,cmap='viridis',s=2)

    return f,[ax, aax, aaax]
