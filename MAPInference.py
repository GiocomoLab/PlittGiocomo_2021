import numpy as np
import scipy as sp
import scipy.ndimage.filters as filters
from scipy.interpolate import interp1d as spline
from sklearn.linear_model import LinearRegression as linreg
from sklearn.linear_model import HuberRegressor as hreg
from matplotlib import pyplot as plt


def gaussian(mu,sigma,x):
    '''radial basis function centered at 'mu' with width 'sigma', sampled at 'x' '''
    return np.exp(-(mu-x)**2/sigma**2)

def unif(mu,sigma,x):
    ''' step function/uniform distribution centered at 'mu' with width 'sigma', sampled at 'x' '''
    return 1*(np.abs(x-mu)<=sigma/2)

def gaussian_dens(mu,sigma,x):
    ''' normalize output of gaussian to be valid distribution'''
    v = gaussian(mu,sigma,x)
    return v/v.sum()

def mult_and_norm(dens1,dens2):
    ''' multiply two different distributions and normalize '''
    post = dens1*dens2
    return post/post.sum()

def G(_g,_f):
    ''' convolve _g with _f twice. _g and _f should be of same shape '''
    return np.convolve(np.convolve(_g,_f,mode='same'),_f,mode='same')


def convert_prior_to_log(prior,samp = np.linspace(-.3,1.3,num=1000),plot=False):
    '''convert a prior distribution in stimulus space to logspace for proper sampling
    Input: prior - arbitrary distribution assumed to be sampled in the range(-.3,1.3)
            samp - values for calculation of log prior, should be the same as sampled values of prior
            plot - whether or not to generate plots
    Returns: log_dens - log space density for use with run_posterior_inference
            log_sampling_spline - spline for sampling values of prior according to probability
            (f,ax) - figure and axis handles
    '''

    # normalize to ensure prior is a proper distribution
    prior = np.convolve(prior,gaussian(.5,.1,samp),mode='same')
    prior/=prior.sum()

    # go to cummulative distribution so that conversion maintains density
    cum_prior = np.cumsum(prior)
    cum_prior[0],cum_prior[-1] = 0,1 # correct for float errors
    cum_spline = spline(samp,cum_prior) # fit a linear interpolation spline so that we can arbitrarily sample
    log_cum_spline = spline(morph_2_logstim(samp),cum_prior)
    log_cum = log_cum_spline(samp) # sample cummulative distribution in log space
    log_sampling_spline = spline(cum_prior,morph_2_logstim(samp))  # create transposed spline for sampling values according to their probability

    #convert cumulative density back to density
    log_dens = log_cum[1:]-log_cum[:-1]
    log_dens = np.append(log_dens,log_dens[-1])
    log_dens /=log_dens.sum()

    if plot:
        f,ax = plt.subplots(1,2)
        for a in range(2):
            ax[a].spines['right'].set_visible(False)
            ax[a].spines['top'].set_visible(False)

        ax[0].plot(samp,cum_prior,label='morph space',color='black')
        ax[0].plot(morph_2_logstim(samp),cum_prior,label='log space',color='red')
        ax[0].legend()
        ax[0].set_ylabel('$F(S)$')
        ax[0].set_xlabel('$S$')

        ax[1].plot(samp,prior,color='black')
        ax[1].plot(morph_2_logstim(samp),log_dens,color='red')
        ax[1].set_ylabel('$P(S)$')
        ax[1].set_xlabel('$S$')
        ax[1].set_yticks([])

        return log_dens, log_sampling_spline, (f,ax)
    else:
        return log_dens, log_sampling_spline

def rare_prior(samp = np.linspace(-.3,1.3,num=1000),plot=False):
    ''' get idealized Rare Morph prior and perform log-transform
    inputs: samp - values at which to sample prior, must be in range [-.3,1.3]
            plot - bool, whether or not to generate plots of prior
    returns: prior - original space prior sampled at samp
            log_dens - log transformed prior sampled at samp (log space)
            sampling_spline - spline for sampling values according to probability
            [h - fig, axis handles]
    '''
    f = unif(.5,.2,samp)
    prior = G(unif(0,.2,samp),f) + G(unif(1,.2,samp),f) + 1E-2
    prior = prior/prior.sum()

    if plot:
        log_dens,sampling_spline,h = convert_prior_to_log(prior,samp=samp,plot=plot)
        return prior, log_dens, sampling_spline, h
    else:
        log_dens,sampling_spline = convert_prior_to_log(prior,samp = samp,plot=plot)
        return prior, log_dens, sampling_spline

def freq_prior(samp = np.linspace(-.3,1.3,num=1000), plot = False):
    ''' get idealized Frequent Morph prior and perform log-transform
    inputs: samp - values at which to sample prior, must be in range [-.3,1.3]
            plot - bool, whether or not to generate plots of prior
    returns: prior - original space prior sampled at samp
            log_dens - log transformed prior sampled at samp (log space)
            sampling_spline - spline for sampling values according to probability
            [h - fig, axis handles]
    '''

    f = unif(.5,.2,samp)
    prior = G(unif(0,.2,samp),f) + G(unif(.25,.2,samp),f) + G(unif(.5,.2,samp),f) + G(unif(.75,.2,samp),f) + G(unif(1,.2,samp),f)+1E-2
    prior =prior/prior.sum()

    if plot:
        log_dens,sampling_spline,h = convert_prior_to_log(prior,plot=plot)
        return prior, log_dens,sampling_spline,h
    else:
        log_dens,sampling_spline = convert_prior_to_log(prior,plot=plot)
        return prior, log_dens, sampling_spline

def get_MAP(log_prior,S_hat,sigma=.3,samp=np.linspace(-.3,1.3,num=1000)):
    '''get MAP estimate under log_prior. likelihood sampled at each value of S_hat.
    posterior and prior samples at values of samp
    inputs: log_prior - log space prior
            S_hat - 1D numpy array of likelihood values to test. assumed to be in native space NOT log space
            sigma - width of likelihood function
            samp - values at which posterior and log_prior are sampled
    returns: np array of MAP estimates for each value of S_hat '''

    return np.array([samp[np.argmax(log_prior*gaussian(s,.3,samp))] for s in morph_2_logstim(S_hat).tolist()])


# whole-module scoped variables needed to use splines in other functions
f0, f1 = 3, 12
_theta = np.linspace(-.1,1.1,num=1000)
_theta_log = np.log(f0*(1-_theta)+f1*_theta)
theta_log = (_theta_log-np.amin(_theta_log))/(np.amax(_theta_log)-np.amin(_theta_log))*1.6 - .3
theta_log[0],theta_log[-1]=-.3,1.3 # hard code the edges to correct float errors
theta = np.linspace(-.3,1.3,num=1000)
morph_2_logstim = spline(theta,theta_log)
logstim_2_morph = spline(theta_log,theta)
