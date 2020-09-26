import numpy as np



def getf(s):
    '''calculate frequency of Unity wall stimuli from uncorrected wall morph value'''
    return s*2.5 + (1-s)*3.5

def gettheta(s):
    '''calculate angle of sine waves in Unity of wall stimuli from uncorrected wall morph value'''
    return (s*60. + (1-s)*10.)*np.pi/180

def xfreq(s):
    '''get horizontal component of wall stimulus frequency'''
    return np.abs(getf(s)*1500/120*(np.cos(gettheta(s) + np.pi/4.)-np.sin(gettheta(s)+np.pi/4.)))

def yfreq(s):
    '''get vertical component of wall stimulus frequency'''
    return np.abs(getf(s)*1500/120*(np.cos(gettheta(s) + np.pi/4.)+np.sin(gettheta(s)+np.pi/4.)))

def ang(x,y):
    '''get wall angle after correcting for aspect ratio
    inputs: x - horizontal frequency
            y - vertical frequency
    outputs: angle in degrees'''
    return np.arctan(x/y)*180/np.pi

def wallmorphx(s):
    '''rescale x frequency to be in range of "morphs" '''
    return 1.2*(xfreq(s)-xfreq(-.1))/(xfreq(1.1)-xfreq(-.1))-.1

def inv_wallmorphx(m):
    return (m+.1)*(xfreq(1.1)-xfreq(-.1))/1.2 + xfreq(-.1)

def wallmorphy(s):
    '''rescale y frequency to be in range of "morphs" '''
    return 1.2*(yfreq(s)-yfreq(-.1))/(yfreq(1.1)-yfreq(-.1))-.1
