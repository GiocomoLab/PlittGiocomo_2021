import pathlib 

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import pickle

import morph_analyses as m

df = m.preprocessing.load_session_db(dir= pathlib.Path(m.params.repo_dir) / 'data' / 'morph_vr_data')
df = df[df['RewardCount']>40]
df = df.sort_values(['MouseName','DateTime','SessionNumber'])
df = df[df["Track"]=="TwoTower_foraging"]


def single_session_data(filename):
    print(filename)
    vr_data = m.preprocessing.behavior_dataframe(filename)
    trial_info, tstarts_, teleports_ = m.utilities.by_trial_info(vr_data)
    
    effective_morph = trial_info['morphs'] + trial_info['wallJitter']
    return np.hstack((m.unity_transforms.wallmorphx(effective_morph[:, np.newaxis]), effective_morph[:,np.newaxis]))

def single_mouse_data(mouse_alias, df_mouse):


    data = {}
    for i in range(df_mouse.shape[0]):
        sess = df_mouse.iloc[i]
        date, num = sess['DateFolder'],sess['SessionNumber']
        filename = m.params.data_dir / 'morph_vr_data' / mouse_alias / date / f'TwoTower_foraging_{num}.sqlite'
        key = f'date, sess:{num}'
        
        data[key] = single_session_data(filename)
        
    return data

def stack_morphs(data):
    return np.vstack([arr for arr in data.values()])

def single_mouse_priors(metadata, df, x=np.linspace(-.3,1.3,num=1000)[np.newaxis,:]):
    
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
            'wallmorph prior': np.mean(m.utilities.gaussian(morph_dat[:,0:1], sigma_prior, x), axis=0),
            'morph prior': np.mean(m.utilities.gaussian(morph_dat[:, 1:], sigma_prior, x), axis=0),                
        }
    return morph_dict

def prior_post(morph_hist_dict,first_last = 'first', sigma_likelihood=.3,x=np.linspace(-.3,1.3,num=1000)[np.newaxis,:]):
    
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
    wallmorph_likelihood = m.utilities.gaussian(x.T, sigma_likelihood, x)
    wallmorph_posterior = wallmorph_prior*wallmorph_likelihood
    wallmorph_posterior /= wallmorph_posterior.sum(axis=1,keepdims=True)
    
    morph_likelihood = m.utilities.gaussian(m.unity_transforms.wallmorphx(x.T), sigma_likelihood, x)
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
    for rf, mice in zip(('rare', 'frequent') , (m.mouse_metadata.rare_mice, m.mouse_metadata.frequent_mice)):
        morph_dict[rf] = {}
        for mouse in mice:
            if rf == 'rare':
                metadata = m.mouse_metadata.rare_sessions[mouse]
            elif rf == 'frequent':
                metadata = m.mouse_metadata.frequent_sessions[mouse]
            else:
                raise Exception("mouse must be in rare or frequent")
            morph_dict[rf][mouse] = single_mouse_priors(metadata, df)


    with open(m.params.data_dir / "morph_hists.pkl", 'wb') as file:
        pickle.dump(morph_dict, file)
