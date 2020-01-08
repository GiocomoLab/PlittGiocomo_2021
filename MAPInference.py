import numpy as np
import scipy as sp
import scipy.ndimage.filters as filters
from scipy.interpolate import interp1d as spline
from astropy.convolution import convolve
from sklearn.linear_model import LinearRegression as linreg
from sklearn.linear_model import HuberRegressor as hreg
import os

from matplotlib import pyplot as plt
