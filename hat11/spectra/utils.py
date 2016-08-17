import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

from specutils.io import read_fits
from specutils import Spectrum1D


from specutils.io import read_fits
from specutils import Spectrum1D


class EchelleSpectrum(object):
    def __init__(self, spectrum_list):
        self.spectrum_list = spectrum_list
        
    @classmethod
    def from_fits(cls, path):
        spectrum_list = read_fits.read_fits_spectrum1d(path)
        return cls(spectrum_list)
    
    def get_order(self, order):
        return self.spectrum_list[order]

    def fit_order(self, spectral_order, polynomial_order):
        spectrum = self.get_order(spectral_order)
        mean_wavelength = spectrum.wavelength.mean()
        fit_params = np.polyfit(spectrum.wavelength - mean_wavelength, 
                                spectrum.flux, polynomial_order)
        return fit_params
    
    def predict_continuum(self, spectral_order, fit_params):
        spectrum = self.get_order(spectral_order)
        mean_wavelength = spectrum.wavelength.mean()
        flux_fit = np.polyval(fit_params, 
                              spectrum.wavelength - mean_wavelength)
        return flux_fit


class EchelleSpectra(object):
    def __init__(self, es_list):
        self.es_list = es_list
    
    @classmethod
    def from_fits(cls, path_list):
        es_list = [EchelleSpectrum.from_fits(path) for path in path_list]
        return cls(es_list)
    
    def sum_order(self, order):
        wavelengths = self.es_list[0].get_order(order).wavelength
        total_flux = np.sum([spectrum.get_order(order).flux 
                             for spectrum in self.es_list], axis=0)
        return Spectrum1D.from_array(wavelengths, total_flux)

    
def plot_spectrum(spectrum, norm=None, ax=None, offset=0, margin=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if norm is None:
        norm = np.ones_like(spectrum.flux)
    elif hasattr(norm, 'flux'): 
        norm = norm.flux
    if margin is None:
        ax.plot(spectrum.wavelength, spectrum.flux/norm + offset, **kwargs)
    else: 
        ax.plot(spectrum.wavelength[margin:-margin], 
                spectrum.flux[margin:-margin]/norm[margin:-margin] + offset, 
                **kwargs)
        
        
def continuum_normalize(target_spectrum, standard_spectrum, polynomial_order):
    
    normalized_spectrum_list = []
    
    for spectral_order in range(len(target_spectrum.spectrum_list)):
        # Extract one spectral order at a time to normalize
        standard_order = standard_spectrum.get_order(spectral_order)
        target_order = target_spectrum.get_order(spectral_order)
        
        # Fit the standard's flux in this order with a polynomial
        fit_params = standard_spectrum.fit_order(spectral_order, polynomial_order)
        standard_continuum_fit = standard_spectrum.predict_continuum(spectral_order, 
                                                                     fit_params)
        
        # Normalize the target's flux with the continuum fit from the standard
        target_continuum_fit = target_spectrum.predict_continuum(spectral_order, 
                                                                 fit_params)
        target_continuum_normalized_flux = target_order.flux/target_continuum_fit
        target_continuum_normalized_flux /= np.median(target_continuum_normalized_flux)

        normalized_target_spectrum = Spectrum1D(target_continuum_normalized_flux, 
                                                target_order.wcs)

        normalized_spectrum_list.append(normalized_target_spectrum)
        
    return EchelleSpectrum(normalized_spectrum_list)
