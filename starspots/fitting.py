"""
Tools for fitting a transit model to the clean "spotless" transits
generated in `clean_lightcurves.ipynb`

Fix period, ecc, w. Use params from fiducial least sq fit in
`datacleaner.TransitLightCurve.fiducial_transit_fit` to seed the run.


MCMC methods here have been adapted to allow input with either
quadratic or nonlinear (four parameter) limb-darkening parameterizations.
"""

from copy import deepcopy

from .systemparams import aRs_i

import emcee
import numpy as np
import batman
import matplotlib.pyplot as plt
import astropy.units as u


# def generate_model_lc_short(times, t0, depth, dur, b, q1, q2, q3=None, q4=None):
#     # LD parameters from Deming 2011 http://adsabs.harvard.edu/abs/2011ApJ...740...33D
#     rp = depth**0.5
#     exp_time = (1*u.min).to(u.day).value # Short cadence
#     params = batman.TransitParams()
#     params.t0 = t0                       #time of inferior conjunction
#     params.per = P                     #orbital period
#     params.rp = rp                      #planet radius (in units of stellar radii)
#
#     params.ecc = e                      #eccentricity
#     params.w = w                      #longitude of periastron (in degrees)
#     a, inc = T14b2aRsi(params.per, dur, b, rp, e, w)
#
#     params.a = a                       #semi-major axis (in units of stellar radii)
#     params.inc = inc #orbital inclination (in degrees)
#
#
#     u1 = 2*np.sqrt(q1)*q2
#     u2 = np.sqrt(q1)*(1 - 2*q2)
#
#     if q3 is None and q4 is None:
#         params.u = [u1, u2]                #limb darkening coefficients
#         params.limb_dark = "quadratic"       #limb darkening model
#
#     else:
#         params.u = [q1, q2, q3, q4]
#         params.limb_dark = "nonlinear"
#
#     m = batman.TransitModel(params, times, supersample_factor=7,
#                             exp_time=exp_time)
#     model_flux = m.light_curve(params)
#     return model_flux

def generate_model_lc_short(times, transit_params, t0=None, depth=None,
                            dur=None, b=None, q1=None, q2=None,
                            per=None):
    """
    Generate a short-cadence model light curve.
    
    times : `numpy.ndarray`
    transit_params : `batman.TransitParams`
    t0 : float
    depth : float
    dur : float
    b : float
    q1 : float
    q2 : float
    per : float
    """
    exp_time = (1*u.min).to(u.day).value
    transit_params_copy = deepcopy(transit_params)

    if t0 is not None:
        transit_params_copy.t0 = t0
    if depth is not None:
        transit_params_copy.rp = depth**0.5
    if dur is not None:
        transit_params_copy.duration = dur
    if b is not None:
        transit_params_copy.b = b
    if q1 is not None and q2 is not None:
        u1 = 2*np.sqrt(q1)*q2
        u2 = np.sqrt(q1)*(1 - 2*q2)
        transit_params_copy.u = [u1, u2]
    if per is not None:
        transit_params_copy.per = per

    aRs, inc = aRs_i(transit_params_copy)
    transit_params_copy.a = aRs
    transit_params_copy.inc = inc

    m = batman.TransitModel(transit_params_copy, times, supersample_factor=7,
                            exp_time=exp_time)
    model_flux = m.light_curve(transit_params_copy)
    return model_flux

#### Tools for fitting spotless transits


def lnlike(theta, x, y, yerr, transit_params):
    model = generate_model_lc_short(x, transit_params, *theta)
    return -0.5*(np.sum((y-model)**2/yerr**2))


def lnprior(theta, transit_params):
    t0, depth, dur, b, q1, q2 = theta
    if (0.001 < depth < 0.005 and 0.05 < dur < 0.15 and 0 < b < 1 and
        transit_params.t0-0.1 < t0 < transit_params.t0+0.1 and
        0.0 < q1 < 1.0 and 0.0 < q2 < 1.0):
        return 0.0
    return -np.inf


def lnprob(theta, x, y, yerr, transit_params):
    lp = lnprior(theta, transit_params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr, transit_params)


def run_emcee(p0, x, y, yerr, n_steps, transit_params, n_threads=4, burnin=0.4, n_walkers=50):
    """Run emcee on the spotless transits"""
    ndim = len(p0)
    nwalkers = n_walkers
    n_steps = int(n_steps)
    burnin = int(burnin*n_steps)
    pos = [p0 + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]

    pool = emcee.interruptible_pool.InterruptiblePool(processes=n_threads)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr,
                                                                  transit_params),
                                    pool=pool)

    sampler.run_mcmc(pos, n_steps)
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    return samples, sampler


## Tools for fitting for the transit ephemeris with fixed transit light curve

def lnlike_ephemeris(theta, x, y, yerr, transit_params):
    fit_t0, fit_p = theta

    model = generate_model_lc_short(x, transit_params, per=fit_p, t0=fit_t0)
    return -0.5*(np.sum((y-model)**2/yerr**2))


def lnprior_ephemeris(theta, transit_params):
    t0, P = theta
    if (transit_params.t0-0.1 < t0 < transit_params.t0+0.1 and
        transit_params.per-0.1 < P < transit_params.per+0.1):
        return 0.0
    return -np.inf


def lnprob_ephemeris(theta, x, y, yerr, transit_params):
    lp = lnprior_ephemeris(theta, transit_params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_ephemeris(theta, x, y, yerr, transit_params)


def run_emcee_ephemeris(p0, x, y, yerr, n_steps, transit_params, n_threads=4,
                        burnin=0.4, n_walkers=20):
    """
    Run emcee to calculate the ephemeris

    bestfit_transit_parameters : list
        depth, duration, b, q1, q2
    """
    ndim = len(p0)
    nwalkers = int(n_walkers)
    n_steps = int(n_steps)
    burnin = int(burnin*n_steps)
    pos = [p0 + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]

    pool = emcee.interruptible_pool.InterruptiblePool(processes=n_threads)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_ephemeris,
                                    args=(x, y, yerr,
                                          transit_params),
                                    pool=pool)

    sampler.run_mcmc(pos, n_steps)
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    return samples, sampler


def plot_triangle(samples):
    import corner
    if samples.shape[1] == 2:
        fig = corner.corner(samples, labels=["$t_0$", "$P$"])
    if samples.shape[1] == 6:
        fig = corner.corner(samples, labels=["$t_0$", r"depth", r"duration",
                                               r"$b$", "$q_1$", "$q_2$"])
    elif samples.shape[1] == 8:
        fig = corner.corner(samples, labels=["$t_0$", r"depth", r"duration",
                                               r"$b$", "$q_1$", "$q_2$", "$q_3$", "$q_4$"])
    plt.show()


def print_emcee_results(samples):
    if samples.shape[1] == 6:
        labels = ["t_0", r"depth", r"duration",  r"b", "q_1", "q_2"]
    elif samples.shape[1] == 8:
        labels = ["t_0", r"depth", r"duration", r"b", "q_1", "q_2",
                  "q_3", "q_4"]

    all_results = ""
    for i, label in enumerate(labels):
        mid, minus, plus = np.percentile(samples[:,i], [50, 16, 84])
        lower = mid - minus
        upper = plus - mid

        if i==0:
            latex_string = "{0}: {{{1}}}_{{-{2:0.6f}}}^{{+{3:0.6f}}} \\\\".format(label, mid, lower, upper)
        elif i==1:
            latex_string = "{0}: {{{1:.5f}}}_{{-{2:.5f}}}^{{+{3:.5f}}} \\\\".format(label, mid, lower, upper)
        elif i==2:
            latex_string = "{0}: {{{1:.04f}}}_{{-{2:.04f}}}^{{+{3:.04f}}} \\\\".format(label, mid, lower, upper)
        elif i==3:
            latex_string = "{0}: {{{1:.03f}}}_{{-{2:.03f}}}^{{+{3:.03f}}} \\\\".format(label, mid, lower, upper)
        elif i==4:
            latex_string = "{0}: {{{1:.2f}}}_{{-{2:.2f}}}^{{+{3:.2f}}} \\\\".format(label, mid, lower, upper)
        elif i==5:
            latex_string = "{0}: {{{1:.2f}}}_{{-{2:.2f}}}^{{+{3:.2f}}} \\\\".format(label, mid, lower, upper)
        elif i==6:
            latex_string = "{0}: {{{1:.2f}}}_{{-{2:.2f}}}^{{+{3:.2f}}} \\\\".format(label, mid, lower, upper)
        elif i==7:
            latex_string = "{0}: {{{1:.2f}}}_{{-{2:.2f}}}^{{+{3:.2f}}} \\\\".format(label, mid, lower, upper)
        all_results += latex_string
    return "$"+all_results+"$"
