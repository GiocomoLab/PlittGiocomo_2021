import numpy as np
import scipy as sp
from scipy import signal
import sklearn as sk
from sklearn import neighbors
from . import preprocessing as pp
from . import utilities as u
from . import UnityTransforms as unity
from matplotlib import pyplot as plt
import pickle
import os
from scipy.interpolate import interp1d as spline
from sklearn.linear_model import HuberRegressor as hreg



def calculate_posterior(prior,morphs,xx_lims = (-.1,1.1), sigma_likelihood=.3,calcZ=False):
    '''
    calculate posterior distribution for different morph values.
    likelihood is assumed to be gaussian with width
    morphs are assumed to be corrected wall morph values

    inputs: prior - [1, N] numpy array with PMF of prior distribution sampled at
                    evenly spaced intervals between xx_lims[0] and xx_lims[1]
            morphs - [M,] numpy array of morph values at which to calculate the posterior
                    i.e. array of means for the likelihood
            xx_lims - 2-tuple of limits over which prior is sampled
            sigma_likelihood - width of likelihood function
    returns: post - [M,N] numpy array of posterior distributions calculated at each morph value
                    each row is a distribution
            xx - [N,] numpy array, interval over which posterior is sampled

    '''

    assert (prior.shape[0]==1), "prior is wrong shape, must be 1 x N"
    xx = np.linspace(xx_lims[0],xx_lims[1],num=prior.size) # samples
    post = prior*u.gaussian(morphs[:,np.newaxis],sigma_likelihood,xx[np.newaxis,:]) # prior x likelihood
    Z = post.sum(axis=1)
    post = post/post.sum(axis=1,keepdims=True) # normalize to be a valid distribution
    if calcZ:
        return post, xx, Z
    else:
        return post, xx

def make_sampling_spline(xx,prob):
    '''
    generate a spline to sample from probability distribution "prob"
    inputs - xx - values over which prob is sampled
            prob - probability mass function
    returns - a spline to sample from prob using a uniform random number generator
    '''
    prob/=prob.sum()  # ensure prior is s distribution
    cum_prob = np.cumsum(prob) # cumulative probability
    cum_prob[0],cum_prob[-1] = 0,1 # make sure cum. distribution is actually 0-1

    return spline(cum_prob,xx)


def simulate_session(prior,morphs, n_neurons=100,n_samps=1,rbf_sigma=.4,alpha = 1., beta = .25,xx_lims = (-.3,1.3)):
    '''
    simulate a set of gamma neurons that encode samples from a posterior distribution
    inputs: prior - [1 , N] numpy array, probability mass funciton sampled evenly over interval specified in xx_lims
                    assumed that this is in corrected wall morph space
            morphs - [M,] numpy array of corrected wall morph values over which to calculate the posterior distribution
            n_neurons - number of neurons in simulation, modeled as cells with gamma activity. The shape parameter is
                    determined by radial basis function tuning for different values of the stimulus. Each cell has a different
                    preferred stimulus
            n_samps - number of samples to draw per trial
            rbf_sigma - width of radial basis functions
            alpha - scaling of radial basis function
            beta - additive offset of radial basis function, beta>=0
            xx_lims - limits over which prior is evenly sampled
    returns: sim_data - [M,N*n_samps] numpy array of neuron activity
            sim_data_sm - [M,M] numpy array, trial x trial similarity matrix for simulated population
            sim_data_sf - [M, ] numpy array, similarity fraction for simulated neurons

    '''

    post, xx = calculate_posterior(prior,morphs,xx_lims=xx_lims) # calculate posterior
    neurons = np.zeros([morphs.shape[0],n_neurons*n_samps]) #allocate for neual activity
    _mun = np.linspace(xx_lims[0],xx_lims[1],num=n_neurons) # evenly space means of radial basis functions along sampling limits
    for trial in range(post.shape[0]):
        samplingspline = make_sampling_spline(xx,post[trial,:]) # make spline to sample from posterior
        post_samples = samplingspline(np.random.rand(n_samps))
        for samp_i in range(post_samples.size):
            neurons[trial,n_neurons*samp_i:n_neurons*(samp_i+1)] = u.gaussian(_mun,rbf_sigma,post_samples[samp_i]) # get activation of radial basis function

    simdata=np.random.gamma(alpha*neurons +beta) # sample gamma distributions

    # trial by trial similarity matrix
    simdata_norm = simdata/np.linalg.norm(simdata,axis=1,ord=2,keepdims=True)
    simdata_sm = np.dot(simdata_norm,simdata_norm.T)

    # calculate similarity fraction
    trial_info = {}
    trial_info['morphs']=np.copy(morphs)
    trial_info['morphs'][morphs<.1]=0
    trial_info['morphs'][morphs>.9]=1
    sf = u.similarity_fraction(simdata,trial_info)

    return simdata, simdata_sm, sf

def simmat_distribution(morphs,prior,nperms=100,n_neurons=100,n_samps=1,rbf_sigma=.4,alpha = 1., beta = .25,xx_lims = (-.1,1.1)):
    '''
    simulate a distribution of sessions for a fixed prior.
    inputs: nperms - integer, number of simulated first_sessions
            see simulate_session docstring for other inputs
    returns: SIMDATA - [nperms,N,M] numpy array of simulated activity for each cell
            SIMDATA_SM -  [nperms,M,M] numpy array of trial x trial cosine similarity matrices
            SF - [nperms,M] numpy array of similarity fractions
    '''
    SIMDATA, SIMDATA_SM, SF = [],[],[]
    for it in range(nperms):
        if it%100 == 0:
            print(it)
        simdata,simdata_sm,sf = simulate_session(prior,morphs,n_samps=n_samps,n_neurons=n_neurons,rbf_sigma = rbf_sigma)
        SIMDATA.append(simdata)
        SIMDATA_SM.append(simdata_sm)
        SF.append(sf)
    return np.array(SIMDATA), np.array(SIMDATA_SM), np.array(SF)

def run_simmat_distributions(sess,rare_prior,freq_prior,nperms=1000,n_samps=1,rbf_sigma=.4,alpha = 1.,
                            beta = .25,xx_lims = (-.1,1.1),basedir = "D:\\Suite2P_Data\\"):
    '''
    load real session data and simulate distribution of sessions assuming rare morph or frequent morph prior.
    uses exact trials shown in session and same number of neurons as recording

    inputs: sess - row from pandas array (behavior.sqlite) including session information
            rare_prior - [1,N] numpy array, rare morph condition probability mass function
                        sampled evenly on interval defined by xx_lims
                        assumed to have been calculated in corrected wall morph space
            freq_prior - [1,N] numpy array, freq morph condition probability mass function
                        same assumptions as rare_prior
            nperms - number of simulations in each condition
            n_samps - number of sampls to draw on each trial
            rbf_sigma - width of radial basis functions
            alpha - scaling of radial basis function
            beta - additive constant for radial basis function
            xx_lims - interval over which priors are sampled
            basedir - base directory for finding session data
    returns: morphs - [# of trials,] numpy array, sorted, uncorrected wall morph values
             S_trial_mat - [# of trials, # of position bins, # of neurons] numpy array of cell activity rates, morph sorted,
             np.dot(S_tmat_norm,S_tmat_norm.T) - [# of trials, # of trials] numpy array, trial x trial population similarity matrices
             u.similarity_fraction(S_trial_mat,trial_info) - [# of trials,] numpy array, similarity fraction for real data
             SIMDATA - dict for rare and freq conditions: [nperms,# of trials, # of neurons*nsamps] numpy array, simulated data
             SIMDATA_SM - dict for rare and freq conditions: [nperms, # of trials, # of trials] numpy array, simulated trial x trial similarity matrices
             SIMDATA_SF - dict for rare and freq conditions: [nperms, # of trials] numpy array, similarity fractions for simulated data

    '''


    # load data
    with open(os.path.join(basedir,sess["MouseName"],"%s_%s_%i.pkl" % (sess["Track"],sess["DateFolder"],sess["SessionNumber"])),'rb') as f:
        data = pickle.load(f)

    S, tstart_inds, teleport_inds,VRDat = np.copy(data["S"]), data["tstart_inds"], data["teleport_inds"],data["VRDat"]
    S[np.isnan(S)]=0
    S = S/1546 #np.percentile(S,95,axis=0,keepdims=True)
    S_trial_mat = u.make_pos_bin_trial_matrices(np.copy(S),data['VRDat']['pos']._values,tstart_inds,
                                                teleport_inds,bin_size=10,mat_only=True)

    trial_info = data['trial_info']
    morphs = trial_info['morphs']+trial_info['wallJitter']
    morphsort = np.argsort(morphs)
    morphs = morphs[morphsort]
    trial_info['morphs'] = trial_info['morphs'][morphsort]

    S_trial_mat = S_trial_mat[morphsort,:,:]
    S_trial_mat = sp.ndimage.filters.gaussian_filter1d(S_trial_mat,2,axis=1)
    S_trial_mat[np.isnan(S_trial_mat)]=0
    S_tmat = S_trial_mat.reshape(S_trial_mat.shape[0],-1)
    S_tmat_norm = S_tmat/(np.linalg.norm(S_tmat,axis=1,ord=2,keepdims=True)+1E-8)

    # run simulations
    SIMDATA,SIMDATA_SM,SIMDATA_SF = {},{},{}
    for prior,name in zip([rare_prior,freq_prior],['rare','freq']):
        print(name)
        sd,sdsm,sdsf = simmat_distribution(unity.wallmorphx(morphs),prior,nperms=nperms,n_neurons=S_trial_mat.shape[-1])
        SIMDATA[name],SIMDATA_SM[name], SIMDATA_SF[name]= sd, sdsm,sdsf
    return trial_info,S_trial_mat, np.dot(S_tmat_norm,S_tmat_norm.T), u.similarity_fraction(S_trial_mat,trial_info), SIMDATA, SIMDATA_SM, SIMDATA_SF


def simulate_session_plot_results(sess,rare_prior,freq_prior,nperms=1000,n_samps=1,rbf_sigma=.4,alpha = 1., beta = .25,xx_lims = (-.1,1.1),
                                out_dir = "D:\\Morph_Results\\figures\\TheoreticalSimMats\\"):
    '''
    Simulate similarity matrices from rare and frequent morph condition. Plot results and save dictionary of results

    inputs: sess - row from pandas array with session information
            nperms - number of simulations in each condition
            n_samps - number of samples drawn for each trial
            rbf_sigma - width of radial basis functions
            alpha - scaling of RBF
            beta - additive constant for RBF
            xx_lims - range over which prior is sampled


    '''

    # set minimal value for hypothesis tests
    epsilon = 1/nperms

    # run simulation
    trial_info,S_trial_mat,simmat, sf, SIMDATA,SIMDATA_SM,SIMDATA_SF = run_simmat_distributions(sess,rare_prior,freq_prior,nperms=nperms,
                                                                                            n_samps=n_samps,
                                                                                            rbf_sigma=rbf_sigma,alpha = alpha, beta = beta,
                                                                                            xx_lims = xx_lims)
    morph = trial_info['morphs']+trial_info['wallJitter']
    # make output directory
    sessdir = os.path.join(out_dir,"%s_%s_%i" % (sess["MouseName"],sess["DateFolder"],sess["SessionNumber"]))
    try:
        os.makedirs(sessdir)
    except:
        pass


    # plot real data
    simmat_z = sp.stats.zscore(simmat.ravel())
    f,ax = plt.subplots(1,2,figsize=[10,5])
    ax[0].imshow(simmat,vmin=np.percentile(simmat,20),vmax=np.percentile(simmat,80),cmap='Greys')
    ax[1].scatter(morphs,1-sf)
    ax[1].set_xlabel('$\hat{S}$')
    ax[1].set_ylabel('SF')
    f.savefig(os.path.join(sessdir,"simmat.pdf"),format='pdf')

    # calculate correlation with rare morph and frequent morph simulations
    simsimmat_rare_z = sp.stats.zscore(SIMDATA_SM['rare'].reshape(SIMDATA_SM['rare'].shape[0],-1),axis=-1)
    simsimmat_freq_z = sp.stats.zscore(SIMDATA_SM['freq'].reshape(SIMDATA_SM['freq'].shape[0],-1),axis=-1)
    corr_rare, corr_freq = np.dot(simsimmat_rare_z,simmat_z)/simmat_z.shape[0],np.dot(simsimmat_freq_z,simmat_z)/simmat_z.shape[0]

    # calculate test statistic - difference in medians
    TSTAT = np.median(corr_rare)-np.median(corr_freq)
    print('median difference',TSTAT)

    # calculate test statistic for each simulated session
    null_dist = {}
    keys = ['rare','freq']
    for ind  in [0,1]:
        # same prior data
        same_sms = SIMDATA_SM[keys[ind]]
        same_sms = sp.stats.zscore(same_sms.reshape(same_sms.shape[0],-1),axis=1)
        # other prior data
        diff_sms = SIMDATA_SM[keys[ind-1]]
        diff_sms = sp.stats.zscore(diff_sms.reshape(diff_sms.shape[0],-1),axis=1)
        null_dist[keys[ind]]=[]
        for row in range(SIMDATA_SM[keys[ind]].shape[0]):
            mask = np.ones((same_sms.shape[0],))
            mask[row]=0
            mask = mask>0

            # calculate correlation
            test,train = same_sms[row,:], same_sms[mask,:]
            corr_same, corr_diff = np.dot(train,test)/test.shape[0],np.dot(diff_sms,test)/test.shape[0]
            # calculate test statistic
            if ind == 0:
                tstat = np.median(corr_same)-np.median(corr_diff)
            else:
                tstat = np.median(corr_diff)-np.median(corr_same)

            if row%100==0:
                print(row,tstat)
            null_dist[keys[ind]].append(tstat)

        null_dist[keys[ind]]=np.array(null_dist[keys[ind]])

    # log [(probability rare morph test statistic< real test statistic)/(probability frequent morph test statistic> real test statistic)]
    llr = np.log10(np.maximum((null_dist['rare']<TSTAT).sum()/null_dist['rare'].size ,epsilon)) - np.log10(np.maximum((null_dist['freq']>TSTAT).sum()/null_dist['freq'].size ,epsilon))
    print('llr',llr)


    with open(os.path.join(sessdir,'simresults.pkl'),'wb') as file:
        pickle.dump({'SIMDATA':SIMDATA,'SIMDATA_SM':SIMDATA_SM,'SIMDATA_SF':SIMDATA_SF,
                     'morphs':morphs,'corr_rare':corr_rare,'corr_freq':corr_freq,'null_dist':null_dist,'llr':llr,'test_statistic':TSTAT},file)

    # plot histogram of test statistics from simulated data
    f,ax = plt.subplots()
    ax.hist(null_dist['freq'],alpha=.3)
    ax.hist(null_dist['rare'],alpha=.3)
    # plot test statistic from real data
    ax.vlines(TSTAT,0,nperms/4)
    ax.set_xlabel('test statistic')
    ax.set_ylabel('count')
    ax.set_title("Med Diff=%.3f, LLR=%.3f" %(TSTAT,llr))
    f.savefig(os.path.join(sessdir,"null_dists.pdf"),format='pdf')
    f.savefig(os.path.join(sessdir,"null_dists.png"),format='png')

    # plot correlation of real data with simulated rare and frequent morph data
    f,ax = plt.subplots()
    allcorrs = np.concatenate((corr_rare,corr_freq))
    edges = np.linspace(np.amin(allcorrs),np.amax(allcorrs),num=25)
    rare_bins,_ = np.histogram(corr_rare,bins=edges)
    freq_bins,_ = np.histogram(corr_freq,bins=edges)
    ax.fill_between(edges[1:],rare_bins,alpha=.3,color='orange')
    ax.fill_between(edges[1:],freq_bins,alpha=.3,color='blue')
    ax.set_xlabel('correlation')
    ax.set_ylabel('count')
    ax.set_title("Med Diff=%.3f, LLR=%.3f" %(TSTAT,llr))
    f.savefig(os.path.join(sessdir,"corr_distributions.pdf"),format='pdf')


    f,ax = plt.subplots()
    ax.plot(edges[1:],np.cumsum(rare_bins)/1000,color='orange')
    ax.plot(edges[1:],np.cumsum(freq_bins)/1000,color='blue')
    f.savefig(os.path.join(sessdir,"corr_cumdistributions.pdf"),format='pdf')

    # plot some example simulated sessions
    for name in ['rare', 'freq']:
        f,ax = plt.subplots(20,3,figsize=[15,100])
        for it in range(20):
            ax[it,0].imshow(SIMDATA[name][it,:,:],aspect='auto',cmap='pink')
            ax[it,0].set_xlabel('neuron index')
            ax[it,0].set_ylabel('trial (sorted)')
            ax[it,0].set_title('single cell activity')

            ax[it,1].imshow(SIMDATA_SM[name][it,:,:],vmin=np.percentile(SIMDATA_SM[name][it,:,:],20),vmax=np.percentile(SIMDATA_SM[name][it,:,:],80),cmap='Greys')
            ax[it,1].set_title('trial x trial similarity')

            ax[it,2].scatter(morphs,1-SIMDATA_SF[name][it,:])
            ax[it,2].set_ylabel("SF")
            ax[it,2].set_xlabel("morph value")
        f.savefig(os.path.join(sessdir,"%s_eg_simulations.pdf" % name),format='pdf')


if __name__ == '__main__':
    df = pp.load_session_db(dir='D:\\')
    df = df[df['RewardCount']>40]
    df = df[df['Imaging']==1]
    df = df.sort_values(['MouseName','DateTime','SessionNumber'])
    df = df[df["Track"]=="TwoTower_foraging"]
    for (mouse, first_ind)  in zip(['4139265.3','4139265.4','4139265.5','4222168.1','4343703.1','4222153.1','4222153.2',
        '4222153.3','4222174.1','4222154.1','4343702.1'],[5,5,5,3,5,4,4,4,4,4,4]):
        df_mouse = df[df["MouseName"]==mouse]

        for sess_ind in range(first_ind,df_mouse.shape[0]):
            sess = df_mouse.iloc[sess_ind]
            simulate_session_plot_results(sess)
