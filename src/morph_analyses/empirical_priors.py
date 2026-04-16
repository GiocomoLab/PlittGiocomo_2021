"""
Empirical prior computation from VR behavioral data.

Reads the mouse's history of VR sessions and estimates the prior distribution
over stimulus values (wall morph) that the animal experienced before each test
session. Two flavors of prior are computed for each mouse:
  - 'wallmorph prior': probability over the rescaled horizontal frequency axis
  - 'morph prior': probability over the raw morph value axis

The module-level code at import time loads and filters the VR session database
(TwoTower_foraging sessions with >40 rewards). When run as __main__, it computes
and pickles the empirical priors for all rare- and frequent-morph mice to
data/morph_hists.pkl, which is consumed by similarity_fraction.py and the
prior/posterior figure notebooks.
"""

import pathlib

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import pickle

from . import preprocessing, params, utilities
from . import unity_transforms, mouse_metadata

df = preprocessing.load_session_db(dir= pathlib.Path(params.repo_dir) / 'data' / 'morph_vr_data')
df = df[df['RewardCount']>40]
df = df.sort_values(['MouseName','DateTime','SessionNumber'])
df = df[df["Track"]=="TwoTower_foraging"]


def single_session_data(filename):
    '''Extract per-trial wall morph and raw morph values from a single VR session.

    Loads the VR sqlite file, computes per-trial effective morph (morphs + wallJitter),
    and returns both the rescaled horizontal wall frequency (wallmorphx) and the
    raw effective morph as a stacked array.

    inputs: filename - path to a VR .sqlite file
    outputs: [ntrials, 2] array where column 0 is wallmorphx(effective_morph)
             and column 1 is effective_morph
    '''
    print(filename)
    vr_data = preprocessing.behavior_dataframe(filename)
    trial_info, tstarts_, teleports_ = utilities.by_trial_info(vr_data)
    
    effective_morph = trial_info['morphs'] + trial_info['wallJitter']
    return np.hstack((unity_transforms.wallmorphx(effective_morph[:, np.newaxis]), effective_morph[:,np.newaxis]))

def single_mouse_data(mouse_alias, df_mouse):
    '''Collect per-trial morph data for all sessions belonging to one mouse.

    Iterates over the rows of df_mouse, constructs the VR sqlite file path from
    the alias and session metadata, and calls single_session_data on each.

    inputs: mouse_alias - string alias matching the directory structure under
                          params.data_dir / 'morph_vr_data' (e.g. '4139265.3')
            df_mouse    - subset of the session database DataFrame for this mouse
    outputs: data       - dict keyed by 'date, sess:<num>' with [ntrials, 2] arrays
    '''


    data = {}
    for i in range(df_mouse.shape[0]):
        sess = df_mouse.iloc[i]
        date, num = sess['DateFolder'],sess['SessionNumber']
        filename = params.data_dir / 'morph_vr_data' / mouse_alias / date / f'TwoTower_foraging_{num}.sqlite'
        key = f'date, sess:{num}'
        
        data[key] = single_session_data(filename)
        
    return data

def stack_morphs(data):
    '''Vertically stack morph arrays from all sessions in a data dict.

    inputs: data - output of single_mouse_data (dict of [ntrials, 2] arrays)
    outputs: [total_trials, 2] stacked array of (wallmorph, raw morph) values
    '''
    return np.vstack([arr for arr in data.values()])

def single_mouse_priors(metadata, df, x=np.linspace(-.3,1.3,num=1000)[np.newaxis,:]):
    '''Compute empirical prior distributions for a single mouse's first and last test session.

    For each test session labelled 'first' (day 8) and 'last' (final test day),
    collects all VR sessions the mouse experienced prior to that test session and
    fits a Gaussian kernel density estimate (sigma=0.1) over the observed morph
    and wall morph values.

    inputs: metadata - mouse metadata dict from mouse_metadata.rare_sessions or
                       mouse_metadata.frequent_sessions (must have 'alias' and
                       'test_sessions' keys)
            df       - full session database DataFrame (from preprocessing.load_session_db),
                       filtered to TwoTower_foraging sessions with >40 rewards
            x        - [1, N] array of stimulus values at which to evaluate the prior
                       (default: linspace(-0.3, 1.3, 1000))
    outputs: morph_dict - dict with keys 'first' and 'last', each containing:
                 'wallmorph prior' - [N,] KDE prior over rescaled wall frequency axis
                 'morph prior'     - [N,] KDE prior over raw morph axis
    '''
    
    alias = metadata['alias']
    test_sessions = metadata['test_sessions']

    mouse_df = df.loc[df["MouseName"]==alias,:]

    morph_dict = {}

    # get datetime of first test session and session
    for first_last, test_sess in zip(('first', 'last'), (test_sessions[0], test_sessions[-1])):
        # print(mouse, alias, test_sess)
        date_time = test_sess['datetime']
        sess_num = test_sess['session']

        # get all behavioral sessions prior
        if first_last=='first':
            prev_days_mask = (mouse_df['DateTime']-mouse_df['DateTime'].iloc[0]).apply(lambda x: x.days)<8   
        else:
            prev_days_mask = (mouse_df['DateTime']-date_time).apply(lambda x: x.total_seconds()) < 0
        
        # print(prev_days_mask)
        same_day_mask = (mouse_df['DateTime']-date_time).apply(lambda x: x.total_seconds()) == 0
        sess_mask = mouse_df['SessionNumber']<sess_num
       
        # print(sess_mask)
        include_mask = prev_days_mask + (same_day_mask*sess_mask)
        # print(include_mask)
        include_df = mouse_df.loc[include_mask,:]   

        morph_dat = stack_morphs(single_mouse_data(alias,include_df)) 

        sigma_prior=.1

        morph_dict[first_last] = {
            'wallmorph prior': np.mean(utilities.gaussian(morph_dat[:,0:1], sigma_prior, x), axis=0),
            'morph prior': np.mean(utilities.gaussian(morph_dat[:, 1:], sigma_prior, x), axis=0),                
        }
    return morph_dict

def prior_post(morph_hist_dict, first_last='first', sigma_likelihood=.3, x=np.linspace(-.3,1.3,num=1000)[np.newaxis,:]):
    '''Average empirical priors across mice and compute Bayesian posteriors.

    Takes the per-mouse prior distributions from morph_hist_dict, normalizes and
    averages them across all mice, then multiplies by a Gaussian likelihood
    (width sigma_likelihood) at each stimulus value to produce per-stimulus
    posterior distributions.

    inputs: morph_hist_dict  - dict of per-mouse prior dicts from single_mouse_priors
                               (keyed by mouse identifier)
            first_last       - 'first' or 'last'; which test session's prior to use
            sigma_likelihood - width of the Gaussian likelihood function (default 0.3)
            x                - [1, N] array of stimulus values (default linspace(-0.3, 1.3, 1000))
    outputs: dict with keys:
                'each_mouse_wallmorph_prior' - [mice, N] unnormalized per-mouse priors (wall morph)
                'combined_wallmorph_prior'   - [1, N] cross-mouse average (wall morph)
                'wallmorph_likelihood'       - [N, N] Gaussian likelihood matrix
                'wallmorph_posterior'        - [N, N] posterior (rows=stimulus, cols=estimate)
                'each_mouse_morph_prior'     - [mice, N] per-mouse priors (raw morph)
                'combined_morph_prior'       - [1, N] cross-mouse average (raw morph)
                'morph_likelihod'            - [N, N] likelihood in raw morph space
                'morph_posterior'            - [N, N] posterior in raw morph space
    '''
    
    mice = [mouse for mouse in morph_hist_dict.keys()]
    
    wallmorph_priors, morph_priors = np.zeros([len(mice),x.shape[1]]),np.zeros([len(mice),x.shape[1]])
    
    for i, (mouse, d) in enumerate(morph_hist_dict.items()):
        wallmorph_priors[i, :] = d[first_last]['wallmorph prior']
        morph_priors[i, :] = d[first_last]['morph prior']
    
    # normalize each mouse to sum to 1
    wallmorph_priors /= wallmorph_priors.sum(axis=-1, keepdims=True)    
    morph_priors /= morph_priors.sum(axis=-1, keepdims=True)
        
    # average across mice
    wallmorph_prior = wallmorph_priors.mean(axis=0, keepdims=True)
    morph_prior = morph_priors.mean(axis=0, keepdims=True)
    # normalize again
    wallmorph_prior /= wallmorph_prior.sum(keepdims=True)
    morph_prior /= morph_prior.sum(keepdims=True)
    
    # calculate posterior
    wallmorph_likelihood = utilities.gaussian(x.T, sigma_likelihood, x)
    wallmorph_posterior = wallmorph_prior*wallmorph_likelihood
    wallmorph_posterior /= wallmorph_posterior.sum(axis=1,keepdims=True)
    
    morph_likelihood = utilities.gaussian(unity_transforms.wallmorphx(x.T), sigma_likelihood, x)
    morph_posterior = morph_prior*morph_likelihood
    morph_posterior /= morph_posterior.sum(axis=1, keepdims=True)
    
    return {
        'each_mouse_wallmorph_prior': wallmorph_priors, 
        'combined_wallmorph_prior': wallmorph_prior,
        'wallmorph_likelihood': wallmorph_likelihood,
        'wallmorph_posterior': wallmorph_posterior,
        'each_mouse_morph_prior': morph_priors,
        'combined_morph_prior': morph_prior,
        'morph_likelihod': morph_likelihood,
        'morph_posterior': morph_posterior,
    }


if __name__=="__main__":
    morph_dict= {}
    for rf, mice in zip(('rare', 'frequent') , (mouse_metadata.rare_mice, mouse_metadata.frequent_mice)):
        morph_dict[rf] = {}
        for mouse in mice:
            if rf == 'rare':
                metadata = mouse_metadata.rare_sessions[mouse]
            elif rf == 'frequent':
                metadata = mouse_metadata.frequent_sessions[mouse]
            else:
                raise Exception("mouse must be in rare or frequent")
            morph_dict[rf][mouse] = single_mouse_priors(metadata, df)


    with open(params.data_dir / "morph_hists.pkl", 'wb') as file:
        pickle.dump(morph_dict, file)
