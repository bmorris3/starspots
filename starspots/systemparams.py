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


def kepler17_params_db():

    k17_params = batman.TransitParams()
    k17_params.b = 0.1045441000
    k17_params.t0 = 2455185.678035                       #time of inferior conjunction

    k17_params.per = 1.4857108000                      #orbital period
    k17_params.rp = 0.0179935208**0.5      #planet radius (in units of stellar radii)
    k17_params.duration = 0.09483

    k17_params.ecc = 0                      #eccentricity
    k17_params.w = 90.                       #longitude of periastron (in degrees)
    a, inc = aRs_i(k17_params)

    k17_params.a = a                       #semi-major axis (in units of stellar radii)
    k17_params.inc = 88.9456000000 #orbital inclination (in degrees)

    k17_params.u = [0.59984, -0.165775, 0.6876732, -0.349944]   #limb darkening coefficients
    k17_params.limb_dark = "nonlinear"       #limb darkening model

    return k17_params
