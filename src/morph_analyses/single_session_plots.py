import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib import gridspec

from . import utilities as u 
from .sess import CA1MorphSession 
from . import similarity_matrix_analysis as sm
from . import unity_transforms as ut
from . import similarity_fraction 







def plot_sessions(mouse,session_list, session_labels):
    
    
    f = plt.figure(figsize=[25,7*len(session_list)])
    gs = gridspec.GridSpec(14*len(session_list),50)
    
    
    for i,(sess_deets,label) in enumerate(zip(session_list,session_labels)):
        try:
            sess = CA1MorphSession.from_nwb(mouse, label, sess_deets['day'])
            # VRDat = sess.vr_data
            S = sess.timeseries['spks_norm']
            S[np.isnan(S)]=0
            
            trial_info = sess.trial_info
            S_trial_mat = sess.trial_matrices['spks_norm']
            S_trial_mat[np.isnan(S_trial_mat)]=0
                
            morphs = trial_info['effective_morph']
            effMorph = (morphs +.1)/1.2
                

            # trial x trial similarity
            S_tmat_norm = np.copy(S_trial_mat).reshape(S_trial_mat.shape[0],-1)
            S_tmat_norm = S_tmat_norm/(np.linalg.norm(S_tmat_norm,ord=2,axis=-1,keepdims=True)+1E-3)
            C = np.matmul(S_tmat_norm,S_tmat_norm.T)

            msort = np.argsort(morphs)
            C_msort = sm._sort_simmat(C,msort)

            vmax = np.percentile(C.ravel(),90)
            vmin = np.percentile(C.ravel(),10)

            cm_ax = f.add_subplot(gs[14*i:14*i+9,:9])
            cm_ax.imshow(C_msort,cmap='Greys',vmin=vmin,vmax=vmax,aspect='auto')
            cm_ax.set_yticks([])
            cm_ax.set_xticks([])
            cm_ax.set_title(f"{mouse}, day {sess_deets['day']} ({S.shape[-1]} neurons, {morphs.shape[0]} trials)", fontsize=20)
            # cm_ax.set_title("%s, session %d (%d neurons, %d trials)" % (mouse, label,S.shape[1],morphs.shape[0]),fontsize=20)

            mm_ax = f.add_subplot(gs[14*i+9:14*i+12,:9])
            mm_ax.scatter(np.arange(effMorph.size),effMorph[msort],c=1-effMorph[msort],cmap='cool')
            mm_ax.set_ylabel("S",fontsize=20)
            mm_ax.set_xlim([0,effMorph.size])
            mm_ax.set_yticks([])
            mm_ax.set_yticks([])



            rtsf = u.rt_similarity_fraction(S_trial_mat,trial_info)
            rtsf_ax = f.add_subplot(gs[14*i+2:14*i+12,10:18])
            for t in range(rtsf.shape[0]):
                rtsf_ax.plot(np.arange(5,450,10),rtsf[t,:],alpha=.3,c=plt.cm.cool(1-effMorph[t]))
            rtsf_ax.spines['top'].set_visible(False)
            rtsf_ax.spines['right'].set_visible(False)
            rtsf_ax.set_ylim([np.nanmax(rtsf.ravel())+.1,np.nanmin(rtsf.ravel())-.1])
            rtsf_ax.set_xlabel("Position (cm.)",fontsize=15)
            rtsf_ax.set_ylabel("$SF$",fontsize=15)



                # similarity fraction
            sf = u.similarity_fraction(S_trial_mat,trial_info)
            sf_ax = f.add_subplot(gs[14*i+2:14*i+12,20:28])
            sf_ax.scatter(ut.xfreq(morphs),sf,color='black') #,c=plt.cm.cool(1-effMorph))
            sf_ax.set_ylim([np.nanmax(sf)+.01,np.nanmin(sf)-.01])
            sf_ax.spines['top'].set_visible(False)
            sf_ax.spines['right'].set_visible(False)
            sf_ax.set_xlabel("$f_h$",fontsize=15)
            sf_ax.set_ylabel("$SF$",fontsize=15)




            yhat = (sf- np.median(sf[trial_info['morphs']==0]))/(np.median(sf[trial_info['morphs']==1])-np.median(sf[trial_info['morphs']==0]))*(1.073-.094) +.094 
            yhat = np.clip(yhat,-.3,1.3)
            H,dkl = similarity_fraction.single_session_kldiv(ut.wallmorphx(morphs),yhat)

            h_ax = f.add_subplot(gs[14*i+2:14*i+12,30:38])
            h_ax.imshow(H,cmap='RdPu',extent=[ut.inv_wallmorphx(-.1),ut.inv_wallmorphx(1.1),ut.inv_wallmorphx(1.3),ut.inv_wallmorphx(-.3)])
            h_ax.set_title("$\Delta D_{KL}$ = %.4f" % (dkl),fontsize=15)
            h_ax.set_xlabel("$\hat{f}_h$", fontsize=15)
            h_ax.set_ylabel("$f_h$",fontsize=15)

            e_ax = f.add_subplot(gs[14*i+2:14*i+12,40:48])
            [w,V]=np.linalg.eig(C)
            order = np.argsort(w)[::-1]
            w = w[order]
            V=V[:,order]
            # project trials onto top 3 eigenvectors
            X =  np.matmul(C,V[:,:3])

            e_ax.scatter(X[:,0],X[:,1],c=1-effMorph,cmap='cool',marker='o',s=150)
            e_ax.spines['top'].set_visible(False)
            e_ax.spines['right'].set_visible(False)
        except:
            print(mouse, sess_deets)
                                
    return f


# def set_ops(change_ops={}):
#     '''set options for single_session_figs'''
#     ops = {'behavior':True,
#         'PCA':True,
#         'place cells':True,
#         'trial simmats':True,
#         # 'savefigs':True,
#         'realtime sf':True}
#     for k,v in change_ops.items():
#         ops[k]=v
#     return ops




# def single_session_figs(sess,
#                         timeseries_str='spks_norm',
#                         ops={},
#                         twotower=False, 
#                         downscale=False):
#     '''plot a bunch of useful
#     inputs: sess: row from pandas array of session metadata
#             can also be a dictionary as long as the following fields are present and valid
#             'data file' - raw VR *.sqlite data file path
#             'scanmat' - Neurolabware .mat file path
#             's2pfolder' - Suite2P output path
#             dir: directory for saving figures
#             dFF: use deconvolved activity rate (False, default) or fluorescence (True)
#             ops: dict, options for which analyses to run and plotting (see set_ops)
#             twotower: bool; whether to plot behavior as TwoTower task (True) or foraging (False)
#     '''


#     # set options
#     ops = set_ops(change_ops=ops)
#     # if ops['savefigs']:
#     #     outdir = os.path.join(dir,sess['MouseName'],"%s_%s_%d" % (sess['Track'],sess['DateFolder'],sess['SessionNumber']))
#     #     try:
#     #         os.makedirs(outdir)
#     #     except:
#     #         print("failed to make path",outdir)


#     # load everything up
#     VRDat = sess.vr_data
#     S = sess.timeseries[timeseries_str]
#     # VRDat, C, S, A = pp.load_scan_sess(sess,fneu_coeff=.7)

#     # if dFF:
#         # S = C/1546 # scaling down for better behavior; 1546 = 10*frame rate
#     # else:
#         # S/=1546 # scaling down for better behavior; 1546 = 10*frame rate
#     S[np.isnan(S)]=0.
#     if downscale:
#         S/=1546


#     # get trial by trial info
#     trial_info, tstart_inds, teleport_inds = sess.trial_info, sess.trial_start_inds, sess.teleport_inds
#     # trial_info, tstart_inds, teleport_inds = u.by_trial_info(VRDat)
#     # S_trial_mat, occ_trial_mat, edges,centers = u.make_pos_bin_trial_matrices(S,
#                                                 # VRDat['pos']._values,tstart_inds,
#                                                 # teleport_inds,bin_size=10,
#                                                 # speed = VRDat['speed']._values)
#     S_trial_mat = sess.trial_matrices[timeseries_str]
#     S_trial_mat[np.isnan(S_trial_mat)]=0
#     S_morph_dict = m.utilities.trial_type_dict(S_trial_mat,trial_info['morphs'])
#     # occ_morph_dict = u.trial_type_dict(occ_trial_mat,trial_info['morphs'])

#     effMorph = trial_info['effective_morph'] #trial_info['morphs']+trial_info['wallJitter']+trial_info['bckgndJitter']+trial_info['towerJitter']
#     reward_pos = trial_info['rzone_entry']
#     reward_pos[np.isnan(reward_pos)]= 480

#     if ops['behavior']:
#         #lick data
#         lick_trial_mat = sess.trial_matrices['licks']
#         centers = sess.trial_matrices['centers']
#         # lick_trial_mat= u.make_pos_bin_trial_matrices(VRDat['lick']._values,
#         #                                                     VRDat['pos']._values,
#         #                                                     VRDat['tstart']._values,
#         #                                                     VRDat['teleport']._values,
#         #                                                     mat_only=True,bin_size=10)
#         lick_morph_dict = m.utilities.trial_type_dict(lick_trial_mat,trial_info['morphs'])
#         max_pos = np.copy(trial_info['max_pos'])
#         max_pos[max_pos>440]=np.nan

#         # plot speed data
#         # speed_trial_mat = u.make_pos_bin_trial_matrices(VRDat['speed']._values,
#         #                                                 VRDat['pos']._values,
#         #                                                 VRDat['tstart']._values,
#         #                                                 VRDat['teleport']._values,
#         #                                                 mat_only=True,bin_size=10)
#         # speed_morph_dict = u.trial_type_dict(speed_trial_mat,trial_info['morphs'])


#         # plot lick and speed rasters depending on which task was run
#         if twotower:
#             f_lick, axarr_lick = m.behavior.behavior_raster_task(lick_trial_mat/np.nanmax(lick_trial_mat.ravel()),
#                                                     centers,effMorph,reward_pos/390.,smooth=False)

#             # f_speed,axarr_speed = b.behavior_raster_task(speed_trial_mat/np.nanmax(speed_trial_mat.ravel()),
#             #                                         centers,effMorph,reward_pos/390.,smooth=False)

#         else:
#             f_lick, axarr_lick = m.behavior.behavior_raster_foraging(lick_trial_mat/np.nanmax(lick_trial_mat.ravel()),
#                                                     centers,effMorph,reward_pos/390.,smooth=False)

#             # f_speed,axarr_speed = b.behavior_raster_foraging(speed_trial_mat/np.nanmax(speed_trial_mat.ravel()),
#         #     #                                         centers,effMorph,reward_pos/390.,smooth=False)
#         # if ops['savefigs']:
#         #     f_lick.savefig(os.path.join(outdir,'licks.pdf'),format='pdf')
#             # f_speed.savefig(os.path.join(outdir,'speed.pdf'),format='pdf')

#     if ops['PCA']:
#         # PCA
#         f_pca,[ax_pca, aax_pca, aaax_pca] = plot_pca(S,VRDat)
#         # if ops['savefigs']:
#         #     f_pca.savefig(os.path.join(outdir,'PCA.pdf'),format='pdf')

#     if ops['place cells']:
#         # find place cells
#         masks, SI = sess.place_cell_info['masks'], sess.place_cell_info['SI']
#         # masks, FR, SI = pc.place_cells_calc(S, VRDat['pos']._values,trial_info,
#         #                 VRDat['tstart']._values, VRDat['teleport']._values,
#         #                 method='bootstrap',correct_only=False,speed=VRDat.speed._values,
#         #                 win_trial_perm=True,morphlist=np.unique(trial_info['morphs']).tolist())

#         # plot place cells by morph
#         f_pc, ax_pc,PC_dict = m.place_cell_analyses.plot_placecells(S_morph_dict,masks)

#         # number in each environment
#         print('morph 0 place cells = %g out of %g , %f ' % (masks[0].sum(), masks[0].shape[0], masks[0].sum()/masks[0].shape[0]))
#         print('morph 1 place cells = %g out of %g, %f' % (masks[1].sum(), masks[1].shape[0], masks[1].sum()/masks[1].shape[0]))

#         # single cell plots
#         f_singlecells = m.place_cell_analyses.plot_top_cells(S_trial_mat,masks,SI,effMorph)

#         # position by morph similarity matrix averaging trials
#         SM = m.similarity_matrix_analysis.morph_simmat(S_morph_dict,corr=False)
#         m=np.unique(trial_info['morphs']).size
#         U  = m.similarity_matrix_analysis.morph_mean_simmat(SM,m)
#         f_SM, ax_SM = m.similarity_matrix_analysis.plot_simmat(SM,m)
#         f_U,ax_U = plt.subplots()
#         ax_U.imshow(U,cmap='pink')

#         # if ops['savefigs']:
#         #     f_pc.savefig(os.path.join(outdir,'placecells.pdf'),format='pdf')
#         #     f_singlecells.savefig(os.path.join(outdir,'singlecells.pdf'),format='pdf')
#         #     f_SM.savefig(os.path.join(outdir,'morphxpos_simmat.pdf'),format='pdf')
#         #     f_U.savefig(os.path.join(outdir,'morph_simmat.pdf'),format='pdf')
#         #     with open(os.path.join(outdir,'pc_masks.pkl'),'wb') as f:
#         #         pickle.dump({'masks':masks},f)

#     if ops['trial simmats']:
#         # trial by trial similarity matrix
#         rmask = trial_info['rewards']==0
#         S_trial_mat[np.isnan(S_trial_mat)]=0
#         # S_trial_mat is trials x positions x neurons
#         S_trial_mat = sp.ndimage.gaussian_filter1d(S_trial_mat,1,axis=1) # smooth position by 1 bin

#         #flatten to be trial x positions*neurons
#         S_tmat = np.reshape(S_trial_mat,[S_trial_mat.shape[0],-1])
#         # divide trials by l2-norm
#         S_tmat = S_tmat/np.linalg.norm(S_tmat,ord=2,axis=-1)[:,np.newaxis]
#         # outer product give trial x trial cosine similarity
#         S_t_rmat = np.matmul(S_tmat,S_tmat.T)
#         # ensure that this is positive (should be true if using deconvolved rate)
#         print("negative similarity inds:", (S_t_rmat<0).ravel().sum())

#         # plot results
#         f_stsm,axtup_stsm = m.similarity_matrix_analysis.plot_trial_simmat(S_t_rmat,trial_info)

#         # plot similarity fraction
#         sf = np.zeros(effMorph.shape) # similarity fraction
#         for trial in range(effMorph.shape[0]): # for each trial
#             # get masks for centroids
#             mask0 = trial_info['morphs']==0
#             mask1 = trial_info['morphs']==1
#             # if current trial is in mask, exclude it
#             if trial_info['morphs'][trial]==0:
#                 mask0[trial]=False
#             elif trial_info['morphs'][trial]==1:
#                 mask1[trial]=False

#             # calculate centroids
#             centroid0, centroid1 = np.nanmean(S_tmat[mask0,:],axis=0), np.nanmean(S_tmat[mask1,:],axis=0)
#             centroid0/np.linalg.norm(centroid0,ord=2)
#             centroid1/np.linalg.norm(centroid1,ord=2)

#             # similarity to two centroids
#             angle0,angle1 = np.dot(S_tmat[trial,:],centroid0),np.dot(S_tmat[trial,:],centroid1)
#             # whole trial similarity fraction
#             sf[trial] = angle0/(angle0+angle1)

#         f_sf,ax_sf = plt.subplots()
#         ax_sf.scatter(effMorph,sf,c=1-effMorph,cmap='cool')
#         ax_sf.scatter(effMorph[rmask],sf[rmask],c='black')

#         # eigendecomposition of trial x trial cosine similarity matrix
#         [w,V]=np.linalg.eig(S_t_rmat)
#         order = np.argsort(w)[::-1]
#         w = w[order]
#         V=V[:,order]
#         # project trials onto top 3 eigenvectors
#         X =  np.matmul(S_t_rmat,V[:,:3])
#         f_embed = plt.figure(figsize=[20,20])
#         ax_embed3d = f_embed.add_subplot(221, projection='3d')
#         ax_embed3d.scatter(X[:,0],X[:,1],X[:,2],c=1-effMorph,cmap='cool')


#         ax_embed3d.scatter(X[rmask,0],X[rmask,1],X[rmask,2],c='black')
#         ax_embed2d = f_embed.add_subplot(222)
#         ax_embed2d.scatter(X[:,0],X[:,1],c=1-effMorph,cmap='cool')
#         ax_embed2d.scatter(X[rmask,0],X[rmask,1],c='black')
#         ax_embed3d = f_embed.add_subplot(223, projection='3d')
#         ax_embed3d.scatter(X[:,0],X[:,1],X[:,2],c=np.arange(X.shape[0]),cmap='viridis')
#         ax_embed2d = f_embed.add_subplot(224)
#         ax_embed2d.scatter(X[:,0],X[:,1],c=np.arange(X.shape[0]),cmap='viridis')

#         # try a nonlinear embedding too
#         lem = sk.manifold.SpectralEmbedding(affinity='precomputed',n_components=3)
#         X = lem.fit_transform(S_t_rmat)

#         f_se = plt.figure(figsize=[20,20])
#         ax_se3d = f_se.add_subplot(221, projection='3d')
#         ax_se3d.scatter(X[:,0],X[:,1],X[:,2],c=1-effMorph,cmap='cool')
#         ax_se3d.scatter(X[rmask,0],X[rmask,1],X[rmask,2],c='black')

#         ax_se2d = f_se.add_subplot(222)
#         ax_se2d.scatter(X[:,0],X[:,1],c=1-effMorph,cmap='cool')
#         ax_se2d.scatter(X[rmask,0],X[rmask,1],c='black')

#         ax_se3d = f_se.add_subplot(223, projection='3d')
#         ax_se3d.scatter(X[:,0],X[:,1],X[:,2],c=np.arange(X.shape[0]),cmap='viridis')
#         ax_se2d = f_se.add_subplot(224)
#         ax_se2d.scatter(X[:,0],X[:,1],c=np.arange(X.shape[0]),cmap='viridis')

#         # if ops['savefigs']:
#         #     f_stsm.savefig(os.path.join(outdir,'trial_simmat.pdf'),format='pdf')
#         #     f_embed.savefig(os.path.join(outdir,'simmat_embed.pdf'),format='pdf')
#         #     f_se.savefig(os.path.join(outdir,'simmat_spectembed.pdf'),format='pdf')
#         #     f_sf.savefig(os.path.join(outdir,'sf.pdf'),format='pdf')

#     if ops['realtime sf']:
#         # plot similarity fraction as a function of position
#         if ops['trial simmats']:
#             pass
#         else:
#             S_trial_mat = sp.ndimage.gaussian_filter1d(S_trial_mat,1,axis=1)


#         sf_bin = np.zeros(S_trial_mat.shape[:-1])
#         # divide population by l2 norm at each position bin
#         S_tmat_norm = S_trial_mat/np.linalg.norm(S_trial_mat,2,axis=-1)[:,:,np.newaxis]
#         for trial in range(S_trial_mat.shape[0]):
#             # cross-validate masks for centroids
#             mask0 = trial_info['morphs']==0
#             mask1 = trial_info['morphs']==1
#             if trial_info['morphs'][trial]==0:
#                 mask0[trial]=False
#             elif trial_info['morphs'][trial]==1:
#                 mask1[trial]=False


#             # calculate centroids
#             centroid_0 = S_trial_mat[mask0,:,:].mean(axis=0)
#             centroid_0/=np.linalg.norm(centroid_0,2,axis=-1)[:,np.newaxis]

#             centroid_1 = S_trial_mat[mask1,:,:].mean(axis=0)
#             centroid_1/=np.linalg.norm(centroid_1,2,axis=-1)[:,np.newaxis]

#             # calculate similarity fraction
#             angle0 = np.diagonal(np.matmul(S_tmat_norm[trial,:,:],centroid_0.T))
#             angle1 = np.diagonal(np.matmul(S_tmat_norm[trial,:,:],centroid_1.T))
#             sf_bin[trial,:]=angle0/(angle1+angle0)


#         f_rtsf,ax_rtsf = plt.subplots()
#         for t in range(sf_bin.shape[0]):
#             ax_rtsf.plot(sf_bin[t,:],c=plt.cm.cool(1-effMorph[t]),alpha=.3)

#         # if ops['savefigs']:
#         #     f_rtsf.savefig(os.path.join(outdir,'rt_sf.pdf'),format='pdf')


#     if ops['trial simmats'] and ops['place cells']:

#         # plot trial x trial similarity results using only place cells
#         cellmask = np.zeros([S.shape[1],])>1
#         for k,v in masks.items(): # masks is from pc.place_cells_calc
#             cellmask = cellmask | v

#         S_trial_mat_pc = S_trial_mat[:,:,cellmask]
#         # flatten, normalize, outer product  as above
#         #        exclude possible end of track problems for scenes where there was a timeout
#         if sess['Track'] in ('TwoTower_noTimeout','TwoTower_Timeout','FreqMorph_Timeout','FreqMorph_Decision'):
#             S_tmat = np.reshape(S_trial_mat_pc[:,:20,:],[S_trial_mat.shape[0],-1])
#         else:
#             S_tmat = np.reshape(S_trial_mat_pc,[S_trial_mat.shape[0],-1])
#         S_tmat = S_tmat/np.linalg.norm(S_tmat,ord=2,axis=-1)[:,np.newaxis]
#         S_t_rmat = np.matmul(S_tmat,S_tmat.T)

#         f_stsm,axtup_stsm = m.similiarity_matrix_analysis.plot_trial_simmat(S_t_rmat,trial_info)

#         sf = np.zeros(effMorph.shape)
#         for trial in range(effMorph.shape[0]):
#             mask0 = trial_info['morphs']==0
#             mask1 = trial_info['morphs']==1
#             if trial_info['morphs'][trial]==0:
#                 mask0[trial]=False
#             elif trial_info['morphs'][trial]==1:
#                 mask1[trial]=False

#             # calculate centroids
#             centroid0, centroid1 = np.nanmean(S_tmat[mask0,:],axis=0), np.nanmean(S_tmat[mask1,:],axis=0)
#             centroid0/np.linalg.norm(centroid0,ord=2)
#             centroid1/np.linalg.norm(centroid1,ord=2)

#             # similarity to two centroids
#             angle0,angle1 = np.dot(S_tmat[trial,:],centroid0),np.dot(S_tmat[trial,:],centroid1)
#             # whole trial similarity fraction
#             sf[trial] = angle0/(angle0+angle1)

#         f_sf,ax_sf = plt.subplots()
#         ax_sf.scatter(effMorph,sf,c=1-effMorph,cmap='cool')
#         ax_sf.scatter(effMorph[rmask],sf[rmask],c='black')

#         [w,V]=np.linalg.eig(S_t_rmat)
#         order = np.argsort(w)[::-1]
#         w = w[order]
#         V=V[:,order]
#         X =  np.matmul(S_t_rmat,V[:,:3])
#         f_embed = plt.figure(figsize=[20,20])
#         ax_embed3d = f_embed.add_subplot(221, projection='3d')
#         ax_embed3d.scatter(X[:,0],X[:,1],X[:,2],c=1-effMorph,cmap='cool')


#         ax_embed3d.scatter(X[rmask,0],X[rmask,1],X[rmask,2],c='black')
#         ax_embed2d = f_embed.add_subplot(222)
#         ax_embed2d.scatter(X[:,0],X[:,1],c=1-effMorph,cmap='cool')
#         ax_embed2d.scatter(X[rmask,0],X[rmask,1],c='black')
#         ax_embed3d = f_embed.add_subplot(223, projection='3d')
#         ax_embed3d.scatter(X[:,0],X[:,1],X[:,2],c=np.arange(X.shape[0]),cmap='viridis')
#         ax_embed2d = f_embed.add_subplot(224)
#         ax_embed2d.scatter(X[:,0],X[:,1],c=np.arange(X.shape[0]),cmap='viridis')

#         lem = sk.manifold.SpectralEmbedding(affinity='precomputed',n_components=3)
#         X = lem.fit_transform(S_t_rmat)

#         f_se = plt.figure(figsize=[20,20])
#         ax_se3d = f_se.add_subplot(221, projection='3d')
#         ax_se3d.scatter(X[:,0],X[:,1],X[:,2],c=1-effMorph,cmap='cool')
#         ax_se3d.scatter(X[rmask,0],X[rmask,1],X[rmask,2],c='black')

#         ax_se2d = f_se.add_subplot(222)
#         ax_se2d.scatter(X[:,0],X[:,1],c=1-effMorph,cmap='cool')
#         ax_se2d.scatter(X[rmask,0],X[rmask,1],c='black')

#         ax_se3d = f_se.add_subplot(223, projection='3d')
#         ax_se3d.scatter(X[:,0],X[:,1],X[:,2],c=np.arange(X.shape[0]),cmap='viridis')
#         ax_se2d = f_se.add_subplot(224)
#         ax_se2d.scatter(X[:,0],X[:,1],c=np.arange(X.shape[0]),cmap='viridis')

#         # if ops['savefigs']:
#         #     f_stsm.savefig(os.path.join(outdir,'trial_simmat_pc.pdf'),format='pdf')
#         #     f_embed.savefig(os.path.join(outdir,'simmat_embed_pc.pdf'),format='pdf')
#         #     f_se.savefig(os.path.join(outdir,'simmat_spectembed_pc.pdf'),format='pdf')
#         #     f_sf.savefig(os.path.join(outdir,'sf_pc.pdf'),format='pdf')

#     return



# def plot_pca(C,VRDat):
#     pca = PCA()
#     trialMask = (VRDat['pos']>0) & (VRDat['pos']<445)
#     print(np.isnan(C).sum(),np.isinf(C).sum())
#     X = pca.fit_transform(C/np.amax(C))

#     print(X.shape)
#     f = plt.figure()
#     axarr = []

#     XX = X[trialMask[:X.shape[0]]]
#     XX = XX[::5,:]
#     morph = VRDat.loc[trialMask,'morph']._values
#     morph = morph[::5]
#     pos = VRDat.loc[trialMask,'pos']._values
#     pos = pos[::5]


#     time = VRDat.loc[trialMask,'time']._values
#     time = time[::5]


#     ax=f.add_subplot(131,projection='3d')
#     s_cxt=ax.scatter(XX[:,0],XX[:,1],XX[:,2],c=morph,cmap='cool',s=2)

#     aax = f.add_subplot(132,projection='3d')
#     s_pos=aax.scatter(XX[:,0],XX[:,1],XX[:,2],c=pos,cmap='magma',s=2)

#     aaax = f.add_subplot(133,projection='3d')
#     s_pos=aaax.scatter(XX[:,0],XX[:,1],XX[:,2],c=time,cmap='viridis',s=2)

#     return f,[ax, aax, aaax]
