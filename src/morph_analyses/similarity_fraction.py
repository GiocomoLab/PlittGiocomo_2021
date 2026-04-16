import numpy as np
import scipy as sp
import pickle



from . import utilities as u
from . import params, empirical_priors
from . import unity_transforms as ut
from . import mouse_metadata
from .sess import CA1MorphSession
from . import similarity_fraction



def single_session_kldiv(_wm,_yh, morph_dict=None):
    
    if morph_dict is None:
        with open(params.data_dir / "morph_hists.pkl", 'rb') as file:
            morph_dict = pickle.load(file)
        
    rare_dists = empirical_priors.prior_post(morph_dict['rare'])
    freq_dists = empirical_priors.prior_post(morph_dict['frequent'])
    x = np.linspace(-.3,1.3,num=1000)

    
    rare_post, freq_post = rare_dists['wallmorph_posterior'], freq_dists['wallmorph_posterior']
    
    wmsm = u.gaussian(_wm[:,np.newaxis,np.newaxis],.1,x[np.newaxis,:,np.newaxis])
    yhsm = u.gaussian(_yh[:,np.newaxis,np.newaxis],.1,x[np.newaxis,np.newaxis,:])
    H = np.sum(wmsm*yhsm,axis=0)
    # Z=H.sum(axis=1)
    H/=H.sum(axis=1,keepdims=True)
    # xmask = (x>=-.1) & (x<=1.1)
    rare_kl,freq_kl = [],[]
    for row in range(H.shape[1]):
        if (x[row]>=.1) and (x[row]<=1.1):
            rare_kl.append(sp.stats.entropy(rare_post[row,:],H[row,:],base=2))
            freq_kl.append(sp.stats.entropy(freq_post[row,:],H[row,:],base=2))
    return H, np.array(rare_kl).mean()-np.array(freq_kl).mean()

        
def sf_to_yhat(sf, morphs):
    # hardcoded values to match posterior
    yhat = (sf- np.median(sf[morphs==0]))/(np.median(sf[morphs==1])-np.median(sf[morphs==0]))*(1.073-.094) +.094
    return np.clip(yhat,-.3, 1.3)


def single_sess_reconstruction(sess, ts_name='spks_norm', morph_dict=None, sigma_likelihood=.3):
    if morph_dict is None:
        with open(params.data_dir / "morph_hists.pkl", 'rb') as file:
            morph_dict = pickle.load(file)
        
    rare_dists = empirical_priors.prior_post(morph_dict['rare'])
    freq_dists = empirical_priors.prior_post(morph_dict['frequent'])
    rare_prior = rare_dists['combined_wallmorph_prior']
    freq_prior = freq_dists['combined_wallmorph_prior']
    rare_post = rare_dists['wallmorph_posterior']
    freq_post = rare_dists['wallmorph_posterior']

    x = np.linspace(-.3,1.3,num=1000)

    wallmorph = ut.wallmorphx(sess.trial_info['effective_morph'])
    
    trial_mat = sess.trial_matrices[ts_name]
    trial_mat[np.isnan(trial_mat)]=0
    sf = u.similarity_fraction(trial_mat, sess.trial_info)
    yhat = sf_to_yhat(sf, sess.trial_info['morphs'])
    
    
    wmsm = u.gaussian(wallmorph[:,np.newaxis,np.newaxis],.1,x[np.newaxis,:,np.newaxis])
    yhsm = u.gaussian(yhat[:,np.newaxis,np.newaxis],.1,x[np.newaxis,np.newaxis,:])
    H = np.sum(wmsm*yhsm,axis=0)
    Z = H.sum(axis=1)
    
    H/=H.sum(axis=1,keepdims=True)
    H_prior =H.sum(axis=0)/(u.gaussian(x,sigma_likelihood, x[:,np.newaxis])/Z[:,np.newaxis]).sum(axis=1)
    H_prior /=H_prior.sum()


     
    
    rare_kl,freq_kl = [],[]
    for row in range(H.shape[1]):
        if (x[row]>=.1) and (x[row]<=1.1):
            rare_kl.append(sp.stats.entropy(rare_post[row,:],H[row,:],base=2))
            freq_kl.append(sp.stats.entropy(freq_post[row,:],H[row,:],base=2))
    dkl = np.array(rare_kl).mean()-np.array(freq_kl).mean()
    
    rare_kl_prior= sp.stats.entropy(rare_prior.ravel(),H_prior,base=2)
    freq_kl_prior = sp.stats.entropy(freq_prior.ravel(),H_prior,base=2)
    dkl_prior = rare_kl_prior - freq_kl_prior
    return H, H_prior, dkl, dkl_prior, yhat


def get_kl_div_summary(sessions, trial_mat_key='spks_norm', morph_dict=None):
    if morph_dict is None:
        with open(params.data_dir / "morph_hists.pkl", 'rb') as file:
            morph_dict = pickle.load(file)
            
    H = 0
    dkl = {}
    for mouse, metadata in sessions.items():
        sess_list = metadata['test_sessions']
        
        if len(sess_list)==0:
            continue
        dkl[mouse] = []
        
        H_mouse  = 0
     
        for i, sess_deets in enumerate(sess_list):
            sess = CA1MorphSession.from_nwb(mouse, 'testing', sess_deets['day'])
            wallmorph = ut.wallmorphx(sess.trial_info['effective_morph'])
            trial_mat = sess.trial_matrices[trial_mat_key]
            trial_mat[np.isnan(trial_mat)]=0
            
            sf = u.similarity_fraction(trial_mat, sess.trial_info)
            yhat = similarity_fraction.sf_to_yhat(sf, sess.trial_info['morphs'])
            
            _H, _dkl = single_session_kldiv(wallmorph, yhat, morph_dict=morph_dict)
            
            H_mouse += _H
            dkl[mouse].append(_dkl)
            
        H_mouse /= i+1
        H_mouse /= H_mouse.sum(axis=1,keepdims=True)
        
        H += H_mouse
        
    H /= H.sum(axis=1, keepdims=True)
        
    return H, dkl