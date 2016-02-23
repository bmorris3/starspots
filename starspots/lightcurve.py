"""
Tool for taking the raw data from MAST and producing cleaned light curves
"""
import os
from glob import glob
import shutil
from copy import deepcopy

from .systemparams import aRs_i, transit_duration

from astropy.io import fits
from astropy.time import Time
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.ndimage import gaussian_filter
import batman


def generate_fiducial_model_lc_short(times, transit_params, t0, depth, b):
    exp_time = (1*u.min).to(u.day).value
    transit_params_copy = deepcopy(transit_params)
    transit_params_copy.t0 = t0
    transit_params_copy.rp = depth**0.5
    transit_params_copy.b = b
    transit_params_copy.duration = transit_duration(transit_params)

    m = batman.TransitModel(transit_params_copy, times, supersample_factor=7,
                            exp_time=exp_time)
    model_flux = m.light_curve(transit_params_copy)
    return model_flux


def generate_very_simple_model_lc_short(times, transit_params, depth):
    exp_time = (1*u.min).to(u.day).value
    transit_params_copy = deepcopy(transit_params)
    transit_params_copy.rp = depth**0.5

    m = batman.TransitModel(transit_params_copy, times, supersample_factor=7,
                            exp_time=exp_time)
    model_flux = m.light_curve(transit_params_copy)
    return model_flux


class LightCurve(object):
    """
    Container object for light curves
    """
    def __init__(self, times=None, fluxes=None, errors=None, quarters=None,
                 name=None, transit_params=None):

        #if len(times) < 1:
        #    raise ValueError("Input `times` have no length.")

        if (isinstance(times[0], Time) and isinstance(times, np.ndarray)):
            times = Time(times)
        elif not isinstance(times, Time):
            times = Time(times, format='jd')

        self.times = times
        self.fluxes = fluxes
        if self.times is not None and errors is None:
            errors = np.zeros_like(self.fluxes) - 1
        self.errors = errors
        if self.times is not None and quarters is None:
            quarters = np.zeros_like(self.fluxes) - 1
        self.quarters = quarters
        self.name = name
        self.transit_params = transit_params

    def plot(self, ax=None, quarter=None, show=True, phase=False,
             plot_kwargs={'color':'b', 'marker':'o', 'lw':0},
             ):
        """
        Plot light curve
        """
        if quarter is not None:
            if hasattr(quarter, '__len__'):
                mask = np.zeros_like(self.fluxes).astype(bool)
                for q in quarter:
                    mask |= self.quarters == q
            else:
                mask = self.quarters == quarter
        else:
            mask = np.ones_like(self.fluxes).astype(bool)

        if ax is None:
            ax = plt.gca()

        if phase:
            x = ((self.times.jd - self.transit_params.t0) /
                 self.transit_params.per % 1)
            x[x > 0.5] -= 1
        else:
            x = self.times.jd

        ax.plot(x[mask], self.fluxes[mask],
                **plot_kwargs)
        ax.set(xlabel='Time' if not phase else 'Phase',
               ylabel='Flux', title=self.name)

        if show:
            plt.show()

    def save_to(self, path, overwrite=False, for_stsp=False):
        """
        Save times, fluxes, errors to new directory ``dirname`` in ``path``
        """
        dirname = self.name
        output_path = os.path.join(path, dirname)
        self.times = Time(self.times)

        if not for_stsp:
            if os.path.exists(output_path) and overwrite:
                shutil.rmtree(output_path)

            if not os.path.exists(output_path):
                os.mkdir(output_path)
                for attr in ['times_jd', 'fluxes', 'errors', 'quarters']:
                    np.savetxt(os.path.join(path, dirname, '{0}.txt'.format(attr)),
                            getattr(self, attr))

        else:
            if not os.path.exists(output_path) or overwrite:
                attrs = ['times_jd', 'fluxes', 'errors']
                output_array = np.zeros((len(self.fluxes), len(attrs)), dtype=float)
                for i, attr in enumerate(attrs):
                    output_array[:, i] = getattr(self, attr)
                np.savetxt(os.path.join(path, dirname+'.txt'), output_array)

    @classmethod
    def from_raw_fits(cls, fits_paths, name=None, transit_params=None):
        """
        Load FITS files from MAST into the LightCurve object
        """
        fluxes = []
        errors = []
        times = []
        quarter = []

        for path in fits_paths:
            data = fits.getdata(path)
            header = fits.getheader(path)
            times.append(data['TIME'] + 2454833.0)
            errors.append(data['SAP_FLUX_ERR'])
            fluxes.append(data['SAP_FLUX'])
            quarter.append(len(data['TIME'])*[header['QUARTER']])

        times, fluxes, errors, quarter = [np.concatenate(i)
                                          for i in [times, fluxes, errors, quarter]]

        mask_nans = np.zeros_like(fluxes).astype(bool)
        for attr in [times, fluxes, errors]:
            mask_nans |= np.isnan(attr)

        times, fluxes, errors, quarter = [attr[-mask_nans]
                                           for attr in [times, fluxes, errors, quarter]]

        return LightCurve(times, fluxes, errors, quarters=quarter, name=name,
                          transit_params=transit_params)

    @classmethod
    def from_dir(cls, path, for_stsp=False):
        """Load light curve from numpy save files in ``dir``"""
        if not for_stsp:
            times, fluxes, errors, quarters = [np.loadtxt(os.path.join(path, '{0}.txt'.format(attr)))
                                               for attr in ['times_jd', 'fluxes', 'errors', 'quarters']]
        else:
            quarters = None
            times, fluxes, errors = np.loadtxt(path, unpack=True)

        if os.sep in path:
            name = path.split(os.sep)[-1]
        else:
            name = path

        if name.endswith('.txt'):
            name = name[:-4]

        return cls(times, fluxes, errors, quarters=quarters, name=name)

    def normalize_each_quarter(self, rename=None, polynomial_order=2, plots=False):
        """
        Use 2nd order polynomial fit to each quarter to normalize the data
        """
        quarter_inds = list(set(self.quarters))
        quarter_masks = [quarter == self.quarters for quarter in quarter_inds]

        for quarter_mask in quarter_masks:

            polynomial = np.polyfit(self.times[quarter_mask].jd,
                                    self.fluxes[quarter_mask], polynomial_order)
            scaling_term = np.polyval(polynomial, self.times[quarter_mask].jd)
            self.fluxes[quarter_mask] /= scaling_term
            self.errors[quarter_mask] /= scaling_term

            if plots:
                plt.plot(self.times[quarter_mask], self.fluxes[quarter_mask])
                plt.show()

        if rename is not None:
            self.name = rename

    def mask_out_of_transit(self, oot_duration_fraction=0.5, flip=False):
        """
        Mask out the out-of-transit light curve based on transit parameters
        """
        # Fraction of one duration to capture out of transit
        params = self.transit_params
        phased = (self.times.jd - params.t0) % params.per
        near_transit = ((phased < params.duration*(0.5 + oot_duration_fraction)) |
                        (phased > params.per - params.duration*(0.5 + oot_duration_fraction)))
        if flip:
            near_transit = -near_transit
        sort_by_time = np.argsort(self.times[near_transit].jd)
        return dict(times=self.times[near_transit][sort_by_time],
                    fluxes=self.fluxes[near_transit][sort_by_time],
                    errors=self.errors[near_transit][sort_by_time],
                    quarters=self.quarters[near_transit][sort_by_time],
                    transit_params=self.transit_params)

    def mask_in_transit(self, oot_duration_fraction=0.25):
        return self.mask_out_of_transit(oot_duration_fraction=oot_duration_fraction,
                                        flip=True)

    def get_transit_light_curves(self, plots=False):
        """
        For a light curve with transits only (returned by get_only_transits),
        split up the transits into their own light curves, return a list of
        `TransitLightCurve` objects
        """
        time_diffs = np.diff(sorted(self.times.jd))
        diff_between_transits = self.transit_params.per/2.
        split_inds = np.argwhere(time_diffs > diff_between_transits) + 1

        if len(split_inds) > 1:

            split_ind_pairs = [[0, split_inds[0][0]]]
            split_ind_pairs.extend([[split_inds[i][0], split_inds[i+1][0]]
                                     for i in range(len(split_inds)-1)])
            split_ind_pairs.extend([[split_inds[-1], len(self.times)]])

            transit_light_curves = []
            counter = -1
            for start_ind, end_ind in split_ind_pairs:
                counter += 1
                if plots:
                    plt.plot(self.times.jd[start_ind:end_ind],
                             self.fluxes[start_ind:end_ind], '.-')

                parameters = dict(times=self.times[start_ind:end_ind],
                                  fluxes=self.fluxes[start_ind:end_ind],
                                  errors=self.errors[start_ind:end_ind],
                                  quarters=self.quarters[start_ind:end_ind],
                                  name=counter,
                                  transit_params=self.transit_params)
                transit_light_curves.append(TransitLightCurve(**parameters))
            if plots:
                plt.show()
        else:
            transit_light_curves = []

        return transit_light_curves

    def get_available_quarters(self):
        return list(set(self.quarters))

    def get_quarter(self, quarter):
        this_quarter = self.quarters == quarter
        return LightCurve(times=self.times[this_quarter],
                          fluxes=self.fluxes[this_quarter],
                          errors=self.errors[this_quarter],
                          quarters=self.quarters[this_quarter],
                          name=self.name + '_quarter_{0}'.format(quarter),
                          transit_params=self.transit_params)

    @property
    def times_jd(self):
        return self.times.jd

    def save_split_at_stellar_rotations(self, path, stellar_rotation_period,
                                        overwrite=False):
        dirname = self.name
        output_path = os.path.join(path, dirname)
        self.times = Time(self.times)

        if os.path.exists(output_path) and overwrite:
            shutil.rmtree(output_path)

        stellar_rotation_phase = ((self.times.jd - self.times.jd[0])*u.day %
                                   stellar_rotation_period ) / stellar_rotation_period
        phase_wraps = np.argwhere(stellar_rotation_phase[:-1] >
                                  stellar_rotation_phase[1:])

        split_times = np.split(self.times.jd, phase_wraps)
        split_fluxes = np.split(self.fluxes, phase_wraps)
        split_errors = np.split(self.errors, phase_wraps)
        split_quarters = np.split(self.quarters, phase_wraps)

        header = "JD Flux Uncertainty Quarter"
        for i, t, f, e, q in zip(range(len(split_times)), split_times,
                                 split_fluxes, split_errors, split_quarters):
            np.savetxt(os.path.join(output_path, 'rotation{:02d}.txt'.format(i)),
                       np.vstack([t, f, e, q]).T, header=header)

    def get_quarterly_maxes(self, flux_upper_limit=4e10, plots=False):
        available_quarters = self.get_available_quarters()
        quarters = [self.get_quarter(q) for q in self.get_available_quarters()]

        if plots:
            fig, ax = plt.subplots(14, 1, figsize=(12, 20))

        quarterly_maxes = {}
        for i, quarter_number, lc in zip(range(len(available_quarters)), available_quarters, quarters):
            fluxes = lc.fluxes[lc.fluxes < flux_upper_limit]
            smoothed_fluxes = gaussian_filter(fluxes, sigma=20)
            quarterly_maxes[quarter_number] = np.max(smoothed_fluxes)

            if plots:
                ax[i].plot(fluxes, '.')
                ax[i].plot(smoothed_fluxes, 'r', lw=2)
                ax[i].axhline(np.max(smoothed_fluxes), ls='--', color='k')
                ax[i].set(ylabel='Quarter {0}'.format(quarter_number),
                          ylim=[0.99*smoothed_fluxes.min(), 1.01*smoothed_fluxes.max()],
                          xticks=[])
        return quarterly_maxes


def get_lightcurves_from_rotation_files(path):
    """Load light curve from numpy savetxt files in directory ``path``"""
    rotation_file_paths = glob(os.path.join(path, 'rotation*.txt'))
    lcs = []
    for f in rotation_file_paths:
        times, fluxes, errors, quarters = np.loadtxt(f, unpack=True)

        if os.sep in path:
            name = path.split(os.sep)[-1]
        else:
            name = path
        lcs.append(LightCurve(times, fluxes, errors, quarters=quarters,
                              name=name))
    return lcs

class TransitLightCurve(LightCurve):
    """
    Container for a single transit light curve
    """
    def __init__(self, times=None, fluxes=None, errors=None, quarters=None,
                 name=None, transit_params=None):
        if isinstance(times[0], Time) and isinstance(times, np.ndarray):
            times = Time(times)
        elif not isinstance(times, Time):
            times = Time(times, format='jd')
        self.times = times
        self.fluxes = fluxes
        self.errors = errors
        self.quarters = quarters
        self.name = name
        self.rescaled = False
        self.transit_params = transit_params

    def fit_linear_baseline(self, cadence=1*u.min, return_near_transit=False,
                            plots=False):
        """
        Find OOT portions of transit light curve using similar method to
        `LightCurve.mask_out_of_transit`, fit linear baseline to OOT
        """
        cadence_buffer = cadence.to(u.day).value
        get_oot_duration_fraction = 0
        phased = ((self.times.jd - self.transit_params.t0) %
                  self.transit_params.per)
        near_transit = ((phased < self.transit_params.duration *
                         (0.5 + get_oot_duration_fraction) + cadence_buffer) |
                        (phased > self.transit_params.per -
                         self.transit_params.duration *
                         (0.5 + get_oot_duration_fraction) - cadence_buffer))

        # Remove linear baseline trend
        order = 1
        linear_baseline = np.polyfit(self.times.jd[-near_transit],
                                     self.fluxes[-near_transit], order)
        linear_baseline_fit = np.polyval(linear_baseline, self.times.jd)

        if plots:
            fig, ax = plt.subplots(1, 2, figsize=(15,6))
            ax[0].axhline(1, ls='--', color='k')
            ax[0].plot(self.times.jd, linear_baseline_fit, 'r')
            ax[0].plot(self.times.jd, self.fluxes, 'bo')
            plt.show()

        if return_near_transit:
            return linear_baseline, near_transit
        else:
            return linear_baseline

    def remove_linear_baseline(self, plots=False, cadence=1*u.min):
        """
        Find OOT portions of transit light curve using similar method to
        `LightCurve.mask_out_of_transit`, fit linear baseline to OOT,
        divide whole light curve by that fit.
        """

        linear_baseline, near_transit = self.fit_linear_baseline(cadence=cadence,
                                                                 return_near_transit=True)
        linear_baseline_fit = np.polyval(linear_baseline, self.times.jd)
        self.fluxes =  self.fluxes/linear_baseline_fit
        self.errors = self.errors/linear_baseline_fit

        if plots:
            fig, ax = plt.subplots(1, 2, figsize=(15,6))
            ax[0].axhline(1, ls='--', color='k')
            ax[0].plot(self.times.jd, self.fluxes, 'o')
            #ax[0].plot(self.times.jd[near_transit], self.fluxes[near_transit], 'ro')
            ax[0].set_title('before trend removal')

            ax[1].set_title('after trend removal')
            ax[1].axhline(1, ls='--', color='k')
            ax[1].plot(self.times.jd, self.fluxes, 'o')
            plt.show()


    def fit_polynomial_baseline(self, order=2, cadence=1*u.min,
                                plots=False):
        """
        Find OOT portions of transit light curve using similar method to
        `LightCurve.mask_out_of_transit`, fit linear baseline to OOT
        """
        cadence_buffer = cadence.to(u.day).value
        get_oot_duration_fraction = 0.01
        phased = (self.times.jd - self.transit_params.t0) % self.transit_params.per
        near_transit = ((phased < self.transit_params.duration *
                         (0.5 + get_oot_duration_fraction) + cadence_buffer) |
                        (phased > self.transit_params.per -
                         self.transit_params.duration *
                         (0.5 + get_oot_duration_fraction) - cadence_buffer))

        # Remove polynomial baseline trend after subtracting the times by its
        # mean -- this improves numerical stability for polyfit
        downscaled_times = self.times.jd - self.times.jd.mean()
        polynomial_baseline = np.polyfit(downscaled_times[-near_transit],
                                         self.fluxes[-near_transit], order)
        polynomial_baseline_fit = np.polyval(polynomial_baseline, downscaled_times)

        if plots:
            fig, ax = plt.subplots(1, 2, figsize=(15,6))
            ax[0].axhline(1, ls='--', color='k')
            ax[0].plot(self.times.jd, polynomial_baseline_fit, 'r')
            ax[0].plot(self.times.jd, self.fluxes, 'bo')
            plt.show()

        return polynomial_baseline_fit


    def subtract_polynomial_baseline(self, plots=False, order=2,
                                     cadence=1*u.min):
        """
        Find OOT portions of transit light curve using similar method to
        `LightCurve.mask_out_of_transit`, fit polynomial baseline to OOT,
        subtract whole light curve by that fit.
        """

        polynomial_baseline_fit = self.fit_polynomial_baseline(cadence=cadence,
                                                               order=order)
        old_fluxes = self.fluxes
        self.fluxes = self.fluxes - polynomial_baseline_fit
        self.errors = self.errors

        if plots:
            fig, ax = plt.subplots(1, 2, figsize=(15,6))
            ax[0].plot(self.times.jd, old_fluxes, 'o')
            ax[0].plot(self.times.jd, polynomial_baseline_fit, 'r')
            ax[0].set_title('before trend removal')

            ax[1].set_title('after trend removal')
            ax[1].axhline(0, ls='--', color='k')
            ax[1].plot(self.times.jd, self.fluxes, 'o')
            plt.show()

    def scale_by_baseline(self, linear_baseline_params):
        if not self.rescaled:
            scaling_vector = np.polyval(linear_baseline_params, self.times.jd)
            self.fluxes *= scaling_vector
            self.errors *= scaling_vector
            self.rescaled = True

    def subtract_add_divide(self, quarterly_max, order=2, plots=False):
        """
        Use Leslie's subtract-add-divide normalization technique.
        """
        self.subtract_polynomial_baseline(order=order, plots=plots)
        self.fluxes += quarterly_max
        self.fluxes /= quarterly_max
        self.errors /= quarterly_max

    def fiducial_transit_fit(self, plots=False, min_method=optimize.fmin,
                             model=generate_very_simple_model_lc_short):
        # Determine cadence:
        typical_time_diff = np.median(np.diff(self.times.jd))*u.day
        exp_long = 30*u.min
        exp_short = 1*u.min
        exp_time = (exp_long if np.abs(typical_time_diff - exp_long) < 1*u.min
                    else exp_short).to(u.day).value

        # [t0, depth, dur, b]
        if model == generate_very_simple_model_lc_short:
            # initial_parameters = [0.003365]
            initial_parameters = [self.transit_params.rp**2]
        else:
            initial_parameters = [self.transit_params.t0,
                                  self.transit_params.rp**2,
                                  self.transit_params.duration,
                                  self.transit_params.b]

        def minimize_this(p, times, fluxes, errors, transit_params):
            return np.sum(((model(times, transit_params, *p) -
                            fluxes)/errors)**2)

        fit_result = min_method(minimize_this, initial_parameters,
                                   args=(self.times.jd, self.fluxes,
                                         self.errors, self.transit_params),
                                   disp=False)
        p = fit_result
        init_model = model(self.times.jd, self.transit_params, *initial_parameters)
        model_flux = model(self.times.jd, self.transit_params, *p)

        if plots:
            plt.plot(self.times.jd, init_model, 'g')
            plt.errorbar(self.times.jd, self.fluxes, self.errors, fmt='.')
            plt.plot(self.times.jd, model_flux, 'r')
            plt.show()

        chi2 = np.sum((self.fluxes - model_flux)**2/self.errors**2)/(len(self.fluxes))
        return p, chi2

    @classmethod
    def from_dir(cls, path):
        """Load light curve from numpy save files in ``path``"""
        times, fluxes, errors, quarters = [np.loadtxt(os.path.join(path, '{0}.txt'.format(attr)))
                                           for attr in ['times_jd', 'fluxes', 'errors', 'quarters']]

        if os.sep in path:
            name = path.split(os.sep)[-1]
        else:
            name = path
        return cls(times, fluxes, errors, quarters=quarters, name=name)


def combine_short_and_long_cadence(short_cadence_transit_light_curves_list,
                                   long_cadence_transit_light_curves_list,
                                   long_cadence_light_curve, name=None):
    """
    Find the linear baseline in the out of transit portions of the long cadence
    light curves in ``long_cadence_transit_light_curves_list``. Scale each
    short cadence light curve by that scaling factor.

    Cut out all transits from the ``long_cadence_light_curve``, and leave
    enough time before/after the first short-cadence points near transit to
    ensure no overlapping exposure times.

    Insert the normalized short cadence light curves from
    ``short_cadence_light_curve_list`` into the time series.
    """
    # Find linear baseline near transits in long cadence
    linear_baseline_params = [transit.fit_linear_baseline(cadence=30*u.min)
                              for transit in long_cadence_transit_light_curves_list]

    # Find the corresponding short cadence transit for each long cadence baseline
    # fit, renormalize that short cadence transit accordingly
    scaled_short_transits = []
    for short_transit in short_cadence_transit_light_curves_list:
        for long_transit, baseline_params in zip(long_cadence_transit_light_curves_list, linear_baseline_params):
            if abs(long_transit.times.jd.mean() - short_transit.times.jd.mean()) < 0.1:
                short_transit.scale_by_baseline(baseline_params)
                scaled_short_transits.append(short_transit)

    # Break out all times, fluxes, errors quarters, and weed out those from the
    # long cadence light curve that overlap with the short cadence data
    all_times = long_cadence_light_curve.times.jd
    all_fluxes = long_cadence_light_curve.fluxes
    all_errors = long_cadence_light_curve.errors
    all_quarters = long_cadence_light_curve.quarters

    remove_mask = np.zeros(len(all_times), dtype=bool)
    short_cadence_exp_time = (30*u.min).to(u.day).value
    for scaled_short_transit in scaled_short_transits:
        min_t = scaled_short_transit.times.jd.min() - short_cadence_exp_time
        max_t = scaled_short_transit.times.jd.max() + short_cadence_exp_time
        overlapping_times = (all_times > min_t) & (all_times < max_t)
        remove_mask |= overlapping_times

    remove_indices = np.arange(len(all_times))[remove_mask]
    all_times, all_fluxes, all_errors, all_quarters = [np.delete(arr, remove_indices)
                                                       for arr in [all_times, all_fluxes, all_errors, all_quarters]]

    # Insert the renormalized short cadence data into the pruned long cadence
    # data, return as `LightCurve` object

    all_times = np.concatenate([all_times] + [t.times.jd for t in scaled_short_transits])
    all_fluxes = np.concatenate([all_fluxes] + [t.fluxes for t in scaled_short_transits])
    all_errors = np.concatenate([all_errors] + [t.errors for t in scaled_short_transits])
    all_quarters = np.concatenate([all_quarters] + [t.quarters for t in scaled_short_transits])

    # Sort by times
    time_sort = np.argsort(all_times)
    all_times, all_fluxes, all_errors, all_quarters = [arr[time_sort]
                                                       for arr in [all_times,
                                                                   all_fluxes,
                                                                   all_errors,
                                                                   all_quarters]]

    transit_params = short_cadence_transit_light_curves_list[0].transit_params
    return LightCurve(times=all_times, fluxes=all_fluxes, errors=all_errors,
                      quarters=all_quarters, name=name,
                      transit_params=transit_params)


def concatenate_light_curves(light_curve_list, name=None):
    """
    Combine multiple transit light curves into one `LightCurve` object
    """
    times = []
    fluxes = []
    errors = []
    quarters = []
    for light_curve in light_curve_list:
        times.append(light_curve.times.jd)
        fluxes.append(light_curve.fluxes)
        errors.append(light_curve.errors)
        quarters.append(light_curve.quarters)
    times, fluxes, errors, quarters = [np.concatenate(i)
                                       for i in [times, fluxes,
                                                 errors, quarters]]

    times = Time(times, format='jd')
    transit_params = light_curve_list[0].transit_params
    return LightCurve(times=times, fluxes=fluxes, errors=errors,
                      quarters=quarters, name=name,
                      transit_params=transit_params)


def concatenate_transit_light_curves(light_curve_list, name=None):
    """
    Combine multiple transit light curves into one `TransitLightCurve` object
    """
    times = []
    fluxes = []
    errors = []
    quarters = []
    for light_curve in light_curve_list:
        times.append(light_curve.times.jd)
        fluxes.append(light_curve.fluxes)
        errors.append(light_curve.errors)
        quarters.append(light_curve.quarters)
    times, fluxes, errors, quarters = [np.concatenate(i)
                                       for i in [times, fluxes,
                                                 errors, quarters]]

    times = Time(times, format='jd')
    transit_params = light_curve_list[0].transit_params
    return TransitLightCurve(times=times, fluxes=fluxes, errors=errors,
                             quarters=quarters, name=name,
                             transit_params=transit_params)
