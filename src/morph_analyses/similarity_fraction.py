"""
Similarity-fraction decoding and KL-divergence analysis.

Takes per-trial similarity fraction values (computed in utilities.py) and
converts them to estimated stimulus values (yhat), then compares the resulting
joint distribution of (true wall morph, decoded morph) against the Bayesian
posteriors derived from empirical priors to compute a KL-divergence metric.

A positive delta-KL ( D_KL(rare || H) - D_KL(freq || H) ) means the neural
decoder is better described by the rare-morph prior than the frequent-morph
prior, consistent with experience-dependent changes in hippocampal coding.
"""

import numpy as np
import scipy as sp
import pickle

from . import utilities as u
from . import params, empirical_priors
from . import unity_transforms as ut
from . import mouse_metadata
from .sess import CA1MorphSession
from . import similarity_fraction



def single_session_kldiv(_wm, _yh, morph_dict=None):
    '''Compute delta-KL divergence between the neural decoder and rare vs. frequent priors.

    Builds the joint distribution H[i,j] = P(wall_morph=x_i, yhat=x_j) by
    smoothing each trial's (wall_morph, yhat) pair with a Gaussian kernel
    (sigma=0.1) and summing across trials, then normalizes rows to produce
    a conditional distribution P(yhat | wall_morph). Computes the mean KL
    divergence of each row against the corresponding row of the rare-morph and
    frequent-morph posteriors (restricted to wall morph in [0.1, 1.1]).

    inputs: _wm        - [ntrials,] array of wall morph values (wallmorphx(effective_morph))
            _yh        - [ntrials,] array of decoded morph estimates (from sf_to_yhat)
            morph_dict - pre-loaded dict from data/morph_hists.pkl; if None, loads from disk
    outputs: H     - [1000, 1000] joint/conditional distribution matrix
             dkl   - float; mean D_KL(rare || H) minus mean D_KL(freq || H) (bits)
    '''
    
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
    '''Rescale similarity fraction to estimated stimulus values in morph space.

    Linearly maps the similarity fraction so that the median SF on morph=0 trials
    aligns to 0.094 and the median SF on morph=1 trials aligns to 1.073 (the
    endpoints of the wallmorphx-transformed stimulus axis), then clips to [-0.3, 1.3].

    inputs: sf     - [ntrials,] similarity fraction (output of utilities.similarity_fraction)
            morphs - [ntrials,] mean morph values used to identify the endpoint trials
    outputs: yhat  - [ntrials,] estimated stimulus values clipped to [-0.3, 1.3]
    '''
    # hardcoded values to match the wallmorphx range [0.094, 1.073]
    yhat = (sf- np.median(sf[morphs==0]))/(np.median(sf[morphs==1])-np.median(sf[morphs==0]))*(1.073-.094) +.094
    return np.clip(yhat,-.3, 1.3)


def single_sess_reconstruction(sess, ts_name='spks_norm', morph_dict=None, sigma_likelihood=.3):
    '''Full prior/posterior reconstruction pipeline for a single session.

    Computes the similarity fraction, converts to yhat, builds the joint
    distribution H, infers an implied prior H_prior by dividing H by the
    likelihood, then computes both the conditional-posterior delta-KL (H vs
    empirical posteriors) and the prior-level delta-KL (H_prior vs empirical
    priors). Useful for Figure 6 / extended data showing how well the neural
    code matches the theoretical Bayesian posterior.

    inputs: sess             - CA1MorphSession with trial_matrices[ts_name] populated
            ts_name          - key in sess.trial_matrices to use (default 'spks_norm')
            morph_dict       - pre-loaded dict from data/morph_hists.pkl; if None, loads from disk
            sigma_likelihood - width of the Gaussian likelihood (default 0.3)
    outputs: H         - [1000, 1000] conditional distribution P(yhat | wall_morph)
             H_prior   - [1000,] implied prior recovered by dividing H by the likelihood
             dkl       - float; conditional delta-KL (bits)
             dkl_prior - float; prior-level delta-KL (bits)
             yhat      - [ntrials,] decoded morph estimates
    '''
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
    '''Aggregate KL-divergence metrics across all test sessions for a set of mice.

    Loops over the sessions dict (structured like mouse_metadata.rare_sessions or
    frequent_sessions), loads each test session from NWB, computes the similarity
    fraction and delta-KL, and accumulates a cross-session-average joint
    distribution H.

    inputs: sessions      - dict of mouse metadata; each value must have 'test_sessions'
                            (list of dicts with 'day' key)
            trial_mat_key - timeseries key to use for similarity fraction (default 'spks_norm')
            morph_dict    - pre-loaded dict from data/morph_hists.pkl; if None, loads from disk
    outputs: H    - [1000, 1000] conditional distribution averaged across all mice and sessions
             dkl  - dict keyed by mouse identifier; list of per-session delta-KL values
    '''
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