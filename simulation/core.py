#!/usr/bin/env python3.6

from .utils import divergence
from scipy.ndimage.filters import laplace
from numpy import sum, power, gradient, atleast_2d

def smoluchowski(t, p, D, mu, E, a_region, d):

    # avalanche count
    a = sum(p * a_region)
#    p[a_region] = 0

    # math
    diff = divergence(atleast_2d(gradient(p, *d))*D, d)
    drift = divergence([mu*p*e for e in E], d)
    dp = diff + drift

    return a,dp


def avalanche_probability(x, P, a_e, a_p):
    P_e,P_h = P
    return np.array([
         (1 - P_e) * a_e(x) * (P_e + P_h + P_e*P_h),
        -(1 - P_h) * a_h(x) * (P_e + P_h + P_e*P_h),
        ])
