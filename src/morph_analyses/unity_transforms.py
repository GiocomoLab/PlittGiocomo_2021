"""
Transforms between raw morph values and Unity VR wall stimulus parameters.

The VR environment uses sinusoidal gratings on the walls whose spatial frequency
(f) and orientation angle (theta) change as a function of the morph value s:
  - s=0 (S=0): f=3.5 Hz, theta=10 deg
  - s=1 (S=1): f=2.5 Hz, theta=60 deg

The horizontal component of the grating frequency (xfreq) is used as the
primary stimulus axis for Bayesian inference because it is monotonically related
to the morph value and can be estimated from the retinal projection.

wallmorphx / wallmorphy rescale xfreq / yfreq to the interval [-0.1, 1.1]
(i.e. the "corrected wall morph" space used in map_inference.py and
simulated_sim_mats.py).
"""

import numpy as np


def getf(s):
    '''Calculate spatial frequency of Unity wall grating from uncorrected wall morph value.

    Linearly interpolates between f=3.5 Hz (s=0) and f=2.5 Hz (s=1).

    inputs: s - morph value(s), scalar or array
    outputs: f in Hz
    '''
    return s*2.5 + (1-s)*3.5

def gettheta(s):
    '''Calculate orientation angle (radians) of Unity wall grating from morph value.

    Linearly interpolates between 10 degrees (s=0) and 60 degrees (s=1).

    inputs: s - morph value(s), scalar or array
    outputs: theta in radians
    '''
    return (s*60. + (1-s)*10.)*np.pi/180

def xfreq(s):
    '''Compute horizontal component of wall grating spatial frequency.

    Projects the grating frequency onto the horizontal axis of the retinal
    image given the grating orientation and screen dimensions (1500 x 120 pixels).

    inputs: s - morph value(s), scalar or array
    outputs: horizontal frequency component (cycles/pixel)
    '''
    return np.abs(getf(s)*1500/120*(np.cos(gettheta(s) + np.pi/4.)-np.sin(gettheta(s)+np.pi/4.)))

def yfreq(s):
    '''Compute vertical component of wall grating spatial frequency.

    inputs: s - morph value(s), scalar or array
    outputs: vertical frequency component (cycles/pixel)
    '''
    return np.abs(getf(s)*1500/120*(np.cos(gettheta(s) + np.pi/4.)+np.sin(gettheta(s)+np.pi/4.)))

def ang(x, y):
    '''Get wall grating angle in degrees after correcting for aspect ratio.

    inputs: x - horizontal frequency component
            y - vertical frequency component
    outputs: angle in degrees
    '''
    return np.arctan(x/y)*180/np.pi

def wallmorphx(s):
    '''Rescale horizontal grating frequency to the corrected wall morph axis [-0.1, 1.1].

    Maps xfreq(s) linearly so that xfreq(-0.1) -> -0.1 and xfreq(1.1) -> 1.1.
    The extended range beyond [0, 1] accommodates per-trial jitter. Used as the
    stimulus axis in Bayesian inference (map_inference.py, simulated_sim_mats.py).

    inputs: s - effective morph value(s) (morphs + wallJitter), scalar or array
    outputs: rescaled wall morph value(s) on [-0.1, 1.1]
    '''
    return 1.2*(xfreq(s)-xfreq(-.1))/(xfreq(1.1)-xfreq(-.1))-.1

def inv_wallmorphx(m):
    '''Invert wallmorphx: convert corrected wall morph back to horizontal frequency.

    inputs: m - corrected wall morph value(s) on [-0.1, 1.1]
    outputs: horizontal grating frequency (cycles/pixel)
    '''
    return (m+.1)*(xfreq(1.1)-xfreq(-.1))/1.2 + xfreq(-.1)

def wallmorphy(s):
    '''Rescale vertical grating frequency to the corrected wall morph axis [-0.1, 1.1].

    Analogous to wallmorphx but for the vertical frequency component.

    inputs: s - effective morph value(s), scalar or array
    outputs: rescaled wall morph value(s) on [-0.1, 1.1]
    '''
    return 1.2*(yfreq(s)-yfreq(-.1))/(yfreq(1.1)-yfreq(-.1))-.1
