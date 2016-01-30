"""
Tools for fitting a transit model to the clean "spotless" transits
generated in `clean_lightcurves.ipynb`

Fix period, ecc, w. Use params from fiducial least sq fit in
`datacleaner.TransitLightCurve.fiducial_transit_fit` to seed the run.


MCMC methods here have been adapted to allow input with either
quadratic or nonlinear (four parameter) limb-darkening parameterizations.
"""

import emcee
import numpy as np
import batman
import matplotlib.pyplot as plt
import astropy.units as u

def T14b2aRsi(P, T14, b, RpRs, eccentricity, omega):
    '''
    Convert from duration and impact param to a/Rs and inclination
    '''
    beta = (1 - eccentricity**2)/(1 + eccentricity*np.sin(np.radians(omega)))
    C = np.sqrt(1 - eccentricity**2)/(1 + eccentricity*np.sin(np.radians(omega)))
    i = np.arctan(beta * np.sqrt((1 + RpRs)**2 - b**2)/(b*np.sin(T14*np.pi/(P*C))))
    aRs = b/(np.cos(i) * beta)
    return aRs, np.degrees(i)

ecosw = 0.261# ? 0.082
esinw = 0.085# ? 0.043
eccentricity = np.sqrt(ecosw**2 + esinw**2)
omega = np.degrees(np.arccos(ecosw/eccentricity))

#ecosw = 0.228
#esinw = 0.056
#ecentricity = np.sqrt(ecosw**2 + esinw**2)
#omega = np.degrees(np.arccos(ecosw/eccentricity))

def generate_model_lc_short(times, t0, depth, dur, b, q1, q2, q3=None, q4=None, P=4.8878018,
                            e=eccentricity, w=omega):
    # LD parameters from Deming 2011 http://adsabs.harvard.edu/abs/2011ApJ...740...33D
    rp = depth**0.5
    exp_time = (1*u.min).to(u.day).value # Short cadence
    params = batman.TransitParams()
    params.t0 = t0                       #time of inferior conjunction
    params.per = P                     #orbital period
    params.rp = rp                      #planet radius (in units of stellar radii)

    params.ecc = e                      #eccentricity
    params.w = w                      #longitude of periastron (in degrees)
    a, inc = T14b2aRsi(params.per, dur, b, rp, e, w)

    params.a = a                       #semi-major axis (in units of stellar radii)
    params.inc = inc #orbital inclination (in degrees)


    u1 = 2*np.sqrt(q1)*q2
    u2 = np.sqrt(q1)*(1 - 2*q2)

    if q3 is None and q4 is None:
        params.u = [u1, u2]                #limb darkening coefficients
        params.limb_dark = "quadratic"       #limb darkening model

    else:
        params.u = [q1, q2, q3, q4]
        params.limb_dark = "nonlinear"

    m = batman.TransitModel(params, times, supersample_factor=7,
                            exp_time=exp_time)
    model_flux = m.light_curve(params)
    return model_flux


def generate_model_lc_short_full(times, depth, dur, b, ecosw, esinw, q1, q2,
                                 q3=None, q4=None, fixed_P=None, fixed_t0=None):
    # LD parameters from Deming 2011 http://adsabs.harvard.edu/abs/2011ApJ...740...33D
    rp = depth**0.5
    exp_time = (1*u.min).to(u.day).value # Short cadence
    params = batman.TransitParams()
    params.t0 = fixed_t0                       #time of inferior conjunction
    params.per = fixed_P                     #orbital period
    params.rp = rp                      #planet radius (in units of stellar radii)

    eccentricity = np.sqrt(ecosw**2 + esinw**2)
    omega = np.degrees(np.arccos(ecosw/eccentricity))
    a, inc = T14b2aRsi(params.per, dur, b, rp, eccentricity, omega)

    params.a = a                       #semi-major axis (in units of stellar radii)
    params.inc = inc #orbital inclination (in degrees)

    params.ecc = eccentricity                      #eccentricity
    params.w = omega                       #longitude of periastron (in degrees)

    u1 = 2*np.sqrt(q1)*q2
    u2 = np.sqrt(q1)*(1 - 2*q2)

    if q3 is None and q4 is None:
        params.u = [u1, u2]                #limb darkening coefficients
        params.limb_dark = "quadratic"       #limb darkening model

    else:
        params.u = [q1, q2, q3, q4]
        params.limb_dark = "nonlinear"

    m = batman.TransitModel(params, times, supersample_factor=7,
                            exp_time=exp_time)
    model_flux = m.light_curve(params)
    return model_flux


#### Tools for fitting spotless transits

def lnlike(theta, x, y, yerr, P):
    model = generate_model_lc_short(x, *theta, P=P)
    return -0.5*(np.sum((y-model)**2/yerr**2))

def lnprior(theta, bestfitt0=2454605.89132):
    if len(theta) == 6:
        t0, depth, dur, b, q1, q2 = theta
        if (0.001 < depth < 0.005 and 0.05 < dur < 0.15 and 0 < b < 1 and
            bestfitt0-0.1 < t0 < bestfitt0+0.1 and 0.0 < q1 < 1.0 and 0.0 < q2 < 1.0):
            return 0.0

    elif len(theta) == 8:
        t0, depth, dur, b, q1, q2, q3, q4 = theta
        if (0.001 < depth < 0.005 and 0.05 < dur < 0.15 and 0 < b < 1 and
            bestfitt0-0.1 < t0 < bestfitt0+0.1 and 0.0 < q1 < 1.0 and 0.0 < q2 < 1.0 and
            0.0 < q3 < 1.0 and 0.0 < q4 < 1.0):
            return 0.0
    return -np.inf

def lnprob(theta, x, y, yerr, P):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr, P)

def run_emcee(p0, x, y, yerr, n_steps, n_threads=4, burnin=0.4, P=4.8878018, n_walkers=50):
    """Run emcee on the spotless transits"""
    ndim = len(p0)
    nwalkers = n_walkers
    n_steps = int(n_steps)
    burnin = int(burnin*n_steps)
    pos = [p0 + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]

    pool = emcee.interruptible_pool.InterruptiblePool(processes=n_threads)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr, P),
                                    pool=pool)

    sampler.run_mcmc(pos, n_steps)
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    return samples, sampler


## Tools for fitting for the transit ephemeris with fixed transit light curve

def lnlike_ephemeris(theta, x, y, yerr, bestfit_transit_parameters):
    depth, dur, b, q1, q2 = bestfit_transit_parameters
    model = generate_model_lc_short(x, theta[0], depth, dur, b, q1, q2, P=theta[1])
    return -0.5*(np.sum((y-model)**2/yerr**2))

def lnprior_ephemeris(theta, bestfitt0=2454605.89132):
    t0, P = theta
    if (bestfitt0-0.1 < t0 < bestfitt0+0.1 and 4.5 < P < 5.5):
        return 0.0
    return -np.inf

def lnprob_ephemeris(theta, x, y, yerr, bestfit_transit_parameters):
    lp = lnprior_ephemeris(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_ephemeris(theta, x, y, yerr, bestfit_transit_parameters)

def run_emcee_ephemeris(p0, x, y, yerr, n_steps, bestfit_transit_parameters, n_threads=4, burnin=0.4, n_walkers=20):
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
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_ephemeris, args=(x, y, yerr, bestfit_transit_parameters),
                                    pool=pool)

    sampler.run_mcmc(pos, n_steps)
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    return samples, sampler



#### Tools for fitting spotless transits with ephemeris fixed

def lnlike_fixed_ephem(theta, x, y, yerr):
    model = generate_model_lc_short_full(x, *theta)
    return -0.5*(np.sum((y-model)**2/yerr**2))

def lnprior_fixed_ephem(theta, bestfitt0=2454605.89132):
    if len(theta) == 6:
        depth, dur, b, q1, q2 = theta
        if (0.001 < depth < 0.005 and 0.05 < dur < 0.15 and 0 < b < 1 and
            0.0 < q1 < 1.0 and 0.0 < q2 < 1.0):
            return 0.0

    elif len(theta) == 8:
        t0, depth, dur, b, q1, q2, q3, q4 = theta
        if (0.001 < depth < 0.005 and 0.05 < dur < 0.15 and 0 < b < 1 and
            bestfitt0-0.1 < t0 < bestfitt0+0.1 and 0.0 < q1 < 1.0 and 0.0 < q2 < 1.0 and
            0.0 < q3 < 1.0 and 0.0 < q4 < 1.0):
            return 0.0
    return -np.inf

def lnprob_fixed_ephem(theta, x, y, yerr):
    lp = lnprior_fixed_ephem(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_fixed_ephem(theta, x, y, yerr)
#
# def run_emcee(p0, x, y, yerr, n_steps, n_threads=4, burnin=0.4):
#     """Run emcee on the spotless transits"""
#     ndim = len(p0)
#     nwalkers = 80
#     n_steps = int(n_steps)
#     burnin = int(burnin*n_steps)
#     pos = [p0 + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
#
#     pool = emcee.interruptible_pool.InterruptiblePool(processes=n_threads)
#     sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_fixed_ephem, args=(x, y, yerr),
#                                     pool=pool)
#
#     sampler.run_mcmc(pos, n_steps)
#     samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
#     return samples, sampler


def plot_triangle(samples):
    import triangle
    if samples.shape[1] == 2:
        fig = triangle.corner(samples, labels=["$t_0$", "$P$"])
    if samples.shape[1] == 6:
        fig = triangle.corner(samples, labels=["$t_0$", r"depth", r"duration",
                                               r"$b$", "$q_1$", "$q_2$"])
    elif samples.shape[1] == 8:
        fig = triangle.corner(samples, labels=["$t_0$", r"depth", r"duration",
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
