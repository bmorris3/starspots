"""
Manage the systems' orbital, planet, stellar parameters.
"""
import batman
import numpy as np


def aRs_i(transit_params):
    """
    Convert from duration and impact param to a/Rs and inclination

    Parameters
    ----------
    transit_params : `batman.TransitParams`
        Transit parameters
    Returns
    -------
    aRs : float
        Semi-major axis in units of stellar radii
    i : float
        Orbital inclination in degrees
    """
    eccentricity = transit_params.ecc
    omega = transit_params.w
    b = transit_params.b
    T14 = transit_params.duration
    P = transit_params.per
    RpRs = transit_params.rp

    # Eccentricity term for b -> a/rs conversion
    beta = (1 - eccentricity**2)/(1 + eccentricity*np.sin(np.radians(omega)))

    # Eccentricity term for duration equation:
    c = (np.sqrt(1 - eccentricity**2) /
         (1 + eccentricity*np.sin(np.radians(omega))))

    i = np.arctan(beta * np.sqrt((1 + RpRs)**2 - b**2) /
                  (b * np.sin(T14*np.pi / (P*c))))
    aRs = b/(np.cos(i) * beta)
    return aRs, np.degrees(i)


def transit_duration(transit_params):
    """
    Calculate transit duration from batman transit parameters object.

    Parameters
    ----------
    transit_params : `batman.TransitParams`
    """
    # Eccentricity term for duration equation:
    c = (np.sqrt(1 - transit_params.ecc**2) /
         (1 + transit_params.ecc*np.sin(np.radians(transit_params.w))))

    return (transit_params.per/np.pi *
            np.arcsin(np.sqrt((1 + transit_params.rp)**2 - transit_params.b**2)/
                      (np.sin(np.radians(transit_params.inc)) *
                       transit_params.a))) * c


def rough_hat11_params():
    # http://exoplanets.org/detail/HAT-P-11_b
    params = batman.TransitParams()
    params.t0 = 2454605.89132                       #time of inferior conjunction
    params.per = 4.8878018                     #orbital period
    params.rp = 0.00332**0.5                      #planet radius (in units of stellar radii)
    params.a = 15.01                       #semi-major axis (in units of stellar radii)
    params.b = 0.35
    params.inc = 88.50

    ecosw = 0.261  # Winn 2010
    esinw = 0.085  # Winn 2010
    eccentricity = np.sqrt(ecosw**2 + esinw**2)
    omega = np.degrees(np.arccos(ecosw/eccentricity))

    params.ecc = eccentricity                      #eccentricity
    params.w = omega                       #longitude of periastron (in degrees)

    params.duration = 0.0982
    params.u = [0.6136, 0.1062]                #limb darkening coefficients
    params.limb_dark = "quadratic"       #limb darkening model
    return params

def hat11_params_morris():
    """
    Transit light curve parameters from Brett for HAT-P-11. Some parameters
    constrained by RVs from Winn et al. 2010 [1]_

    Returns
    -------
    params : `~batman.TransitParams`
        Transit parameters for HAT-P-11

    .. [1] http://adsabs.harvard.edu/abs/2010ApJ...723L.223W
    """
    ecosw = 0.261  # Winn et al. 2010
    esinw = 0.085  # Winn et al. 2010
    eccentricity = np.sqrt(ecosw**2 + esinw**2)
    omega = np.degrees(np.arctan2(esinw, ecosw))

    params = batman.TransitParams()
    params.t0 = 2454605.89159720   # time of inferior conjunction
    params.per = 4.88780233        # orbital period
    params.rp = 0.00343**0.5       # planet radius (in units of stellar radii)
    params.b = 0.127                      # impact parameter
    dur = 0.0982                   # transit duration
    params.inc = 89.4234790468     # orbital inclination (in degrees)

    params.ecc = eccentricity      # eccentricity
    params.w = omega              # longitude of periastron (in degrees)
    params.a = 14.7663717          # semi-major axis (in units of stellar radii)
    params.u = [0.5636, 0.1502]    # limb darkening coefficients
    params.limb_dark = "quadratic" # limb darkening model

    # Required by some friedrich methods below but not by batman:
    params.duration = dur
    params.lam = 106.0          # Sanchis-Ojeda & Winn 2011 (soln 1)
    params.inc_stellar = 80     # Sanchis-Ojeda & Winn 2011 (soln 1)
    params.per_rot = 29.984     # Morris periodogram days

    # params.lam = 121.0            # Sanchis-Ojeda & Winn 2011 (soln 2)
    # params.inc_stellar = 168.0    # Sanchis-Ojeda & Winn 2011 (soln 2)
    return params

def k17_params_morris():
    """
    Transit light curve parameters from Brett for Kepler-17. Some parameters
    constrained by Desert et al. (2011) [1]_

    Returns
    -------
    params : `~batman.TransitParams`
        Transit parameters for Kepler-17

    .. [1] http://adsabs.harvard.edu/abs/2011ApJS..197...14D
    """
    sqrt_e_cosw = 0.008
    sqrt_e_sinw = -0.084
    eccentricity = np.sqrt(sqrt_e_cosw**2 + sqrt_e_sinw**2)**2
    omega = np.degrees(np.arctan2(sqrt_e_sinw, sqrt_e_cosw))

    params = batman.TransitParams()
    params.t0 = 2455185.67863     # time of inferior conjunction
    params.per = 1.48571118         # orbital period
    params.rp = 0.01732**0.5            # planet radius (in units of stellar radii)
    params.b = 0.115                      # impact parameter
    dur = 0.0948                  # transit duration
    params.inc = 88.92684              # orbital inclination (in degrees)

    params.ecc = eccentricity      # eccentricity
    params.w = omega               # longitude of periastron (in degrees)
    params.a = 5.661768                # semi-major axis (in units of stellar radii)
    params.u = [0.40368, 0.25764]      # limb darkening coefficients
    params.limb_dark = "quadratic" # limb darkening model

    # Required by some friedrich methods below but not by batman:
    params.duration = dur
    params.lam = 0.0
    params.inc_stellar = 90
    params.per_rot = 12.04
    return params

