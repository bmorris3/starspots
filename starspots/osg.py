import numpy as np
import os
from glob import glob
from scipy.ndimage import gaussian_filter
import shutil
import stat

python_executable = 'python'
falconer_path = '/home/bmorris/git/hat-11/osg/falconer.py'

config_file_text = """
getenv = true
Notification = never
Executable = {executable}
Initialdir = {initial_dir}
Universe = vanilla

Log = {log_file_path}
Error = {error_file_path}
Output = {output_file_path}
"""

def get_transit_parameters(parameter_file_path):
    """
    Load transit parameters from JRAD's file
    """
    # BJDREF = 2454833.
    # params = np.loadtxt(parameter_file_path, unpack=True, skiprows=4, usecols=[0])
    # p_orb = str(params[0])
    # t_0 = str(params[1] - BJDREF)
    # tdur = str(params[2])
    # p_rot = str(params[3])
    # Rp_Rs = str(params[4]**2.)
    # impact = str(params[5])
    # incl_orb = str(params[6])
    # Teff = str(params[7])
    # sden = str(params[8])
    # return p_orb, t_0, tdur, p_rot, Rp_Rs, impact, incl_orb, Teff, sden
    BJDREF = 2454833.
    params = np.loadtxt(parameter_file_path, unpack=True, skiprows=4, usecols=[0])
    p_orb = params[0]
    t_0 = params[1] - BJDREF
    tdur = params[2]
    p_rot = params[3]
    Rp_Rs = params[4]**2.
    impact = params[5]
    incl_orb = params[6]
    Teff = params[7]
    sden = params[8]
    return p_orb, t_0, tdur, p_rot, Rp_Rs, impact, incl_orb, Teff, sden

class STSPRun(object):
    def __init__(self, parameter_file_path=None, light_curve_path=None,
                 executable_path=None, output_dir_path=None, initial_dir=None,
                 condor_config_path=None, n_restarts=None,
                 planet_properties=None, stellar_properties=None,
                 spot_properties=None,
                 action_properties=None):
        """
        Parameters
        ----------
        parameter_file_path : str
            Path to the parameter file with transit properties

        light_curve_path : str
            Path to the raw input light curve (whitened Kepler data)

        executable_path : str
            Path to the python executable file that will be written by this
            object

        output_dir_path : str
            Path to the outputs (*.in, *.dat files and their results)

        initial_dir : str
            Path to the directory where condor jobs will start

        condor_config_path : str
            Path to the condir .cfg file that will be written by this object

        n_steps : int
            Number of steps per small MCMC chain segment

        n_restarts : int
            Number of times to repeat the small MCMC chain segments

        n_walkers : int (even)
            Number of MCMC walkers

        """
        self.parameter_file_path = parameter_file_path
        self.light_curve_path = light_curve_path
        self.executable_path = executable_path
        self.output_dir_path = output_dir_path
        self.initial_dir = initial_dir
        self.condor_config_path = condor_config_path
        self.planet_properties = planet_properties
        self.stellar_properties = stellar_properties
        self.spot_properties = spot_properties
        self.action_properties = action_properties
        self.n_restarts = n_restarts

        if not self.action_properties['n_chains'] % 2 == 0:
            raise ValueError("Number of walkers must be even, got {0}"
                             .format(self.action_properties['n_chains']))

        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)


    def get_transit_parameters(self):
        p_orb = self.planet_properties['period']
        t_0 = self.planet_properties['first_mid_transit_time']
        tdur = self.planet_properties['transit_duration_days']
        p_rot = self.stellar_properties['stellar_rotation_period']
        Rp_Rs = self.planet_properties['transit_depth']
        impact = self.planet_properties['impact_parameter']
        incl_orb = self.planet_properties['inclination']
        Teff = self.stellar_properties['stellar_temperature']
        sden =  self.stellar_properties['mean_stellar_density']
        return p_orb, t_0, tdur, p_rot, Rp_Rs, impact, incl_orb, Teff, sden
    #
    # def get_downsampled_data(self, FLATMODE=False):
    #     """
    #     Load the whitened Kepler data, downsample it out of transit.
    #
    #     Parameters
    #     ----------
    #     FLATMODE : bool (optional)
    #         False by default because: what does this do? (TODO)
    #     """
    #     # where is the original data file to analyze?
    #     tt0, ff0, ee0 = np.loadtxt(self.light_curve_path, unpack=True)
    #
    #     p_orb, t_0, tdur, p_rot, Rp_Rs, impact, incl_orb, Teff, sden = self.get_transit_parameters()
    #     toff = float(tdur) / float(p_orb) # the transit duration in % (phase) units
    #
    #     # down-sample the out-of-transit data
    #     # define the region +/- 1 transit duration for in-transit
    #     OOT = np.where( ((((tt0-float(t_0)) % float(p_orb))/float(p_orb)) > toff) &
    #                     ((((tt0-float(t_0)) % float(p_orb))/float(p_orb)) < 1-toff) )
    #     ITT = np.where( ((((tt0-float(t_0)) % float(p_orb))/float(p_orb)) <= toff) |
    #                     ((((tt0-float(t_0)) % float(p_orb))/float(p_orb)) >= 1-toff) )
    #
    #     # the factor to down-sample by
    #     Nds = 30 # for Kepler 17
    #     # Nds = 10 # for Joe's data
    #
    #     # down sample out-of-transit, grab every N'th data point
    #     OOT_ds = OOT[0][np.arange(0,np.size(OOT), Nds)]
    #
    #     if FLATMODE is True:
    #         # just use in-transit data
    #         idx = ITT[0][0:]
    #         idx.sort()
    #     elif FLATMODE is False:
    #         # use in- and out-of-transit data
    #         idx = np.concatenate((ITT[0][0:], OOT_ds))
    #         idx.sort()
    #
    #     # these arrays are the final, down-sampled data
    #     tt = tt0[idx]
    #     ff = ff0[idx]
    #     ee = ee0[idx]
    #
    #     return tt, ff, ee

    # def make_condor_config(self):
    #     """
    #     Write a condor config file.
    #
    #     Parameters
    #     ----------
    #     condor_config_path : str
    #         Path to config file to create
    #
    #     executable_path : str
    #         Path to executable shell script
    #
    #     initial_dir : str
    #         Path to directory to work in
    #     """
    #     with open(self.condor_config_path, 'w') as condor_config:
    #
    #         condor_config.write(config_file_text.format(log_file_path=os.path.join(self.initial_dir,
    #                                                                     'log.txt'),
    #                                          error_file_path=os.path.join(self.initial_dir,
    #                                                                       'err.txt'),
    #                                          output_file_path=os.path.join(self.initial_dir,
    #                                                                        'out.txt'),
    #                                          initial_dir=self.initial_dir,
    #                                          executable=self.executable_path).lstrip())
    #         #for window in self.window_dirs:
    #         #    for restart in range(self.n_restarts):
    #         n_jobs_to_queue = int(1.7*len(self.window_dirs)*self.n_restarts)
    #         for i in range(n_jobs_to_queue):
    #             condor_config.write("Arguments = {0}\nQueue\n".format(i))
    #
    #     condor_wrapper_path = self.condor_config_path.split('.condor')[0] + '.csh'
    #     with open(condor_wrapper_path, 'w') as wrapper:
    #         # MUST have newline character at the end of the CSH file
    #         wrapper.write("#!/bin/csh\n{0} {1} $1\n".format(python_executable, falconer_path))
    #
    #     #permissions = stat.S_IRWXU + stat.S_IXOTH + stat.S_IROTH
    #     permissions = stat.S_IRWXO + stat.S_IRWXG + stat.S_IRWXU
    #     os.chmod(condor_wrapper_path, permissions)
    #     os.chmod(self.condor_config_path, permissions)
    #
    # def write_data_files(self):
    #     """
    #     Create the *.dat data files for each time window
    #     """
    #     p_orb, t_0, tdur, p_rot, Rp_Rs, impact, incl_orb, Teff, sden = self.get_transit_parameters()
    #     tt, ff, ee = self.get_downsampled_data(FLATMODE=False)
    #
    #     ddur = float(p_rot) * 0.9
    #     # ddur = float(p_rot) * 1.2
    #
    #     # of time windows, shift each window by 1/2 window size
    #     ntrials = np.floor((np.max(tt)-np.min(tt))*2.0/float(p_rot))
    #     #nspot_list = ['8']
    #     #nspots = np.tile(nspot_list, ntrials)
    #     nspot_list = [str(self.spot_properties['n_spots'])]
    #     dstart_all = np.repeat(np.min(tt) + float(p_rot)/2.*np.arange(ntrials),
    #                            len(nspot_list))
    #
    #
    #     # main loop for .in file writing of each time window & #spots
    #     for n in range(len(dstart_all)):
    #         dstart = dstart_all[n]
    #         npts = np.sum((tt >= dstart) & (tt <= dstart+ddur))
    #         wndw = np.where((tt >= dstart) & (tt <= dstart+ddur)) # this could be combined w/ above...
    #
    #         # only run STSP on line if more than N epoch of data
    #         if npts >= 500:
    #
    #             window_dir = os.path.join(self.output_dir_path,
    #                                       "window{0:03d}".format(n))
    #             data_path = os.path.join(window_dir,
    #                                      "window{0:03d}.dat".format(n))
    #             if not os.path.exists(window_dir):
    #                 os.mkdir(window_dir)
    #
    #             # write small chunk of LC to operate on
    #             dfn = open(data_path, 'w')
    #             for k in range(0,npts-1):
    #                 dfn.write(str(tt[wndw[0][k]]) + ' ' +
    #                           str(ff[wndw[0][k]]) + ' ' +
    #                           str(ee[wndw[0][k]]) + '\n')
    #             dfn.close()

    def copy_data_files(self, transit_paths):
        """
        Create the *.dat data files for each time window, using time windows that cover only one transit
        """
        p_orb, t_0, tdur, p_rot, Rp_Rs, impact, incl_orb, Teff, sden = self.get_transit_parameters()

        for transit_path in transit_paths:
            n = int(transit_path.split("transit")[-1].split(".")[0])
            window_dir = os.path.join(self.output_dir_path,
                                      "window{0:03d}".format(n))
            data_path = os.path.join(window_dir,
                                     "window{0:03d}.dat".format(n))
            if not os.path.exists(window_dir):
                os.mkdir(window_dir)

            shutil.copyfile(transit_path, data_path)

        # tt, ff, ee = self.get_downsampled_data(FLATMODE=False)
        #
        # ddur = float(p_rot) * 0.9
        # # ddur = float(p_rot) * 1.2
        #
        # # of time windows, shift each window by 1/2 window size
        # ntrials = np.floor((np.max(tt)-np.min(tt))*2.0/float(p_rot))
        # #nspot_list = ['8']
        # #nspots = np.tile(nspot_list, ntrials)
        # nspot_list = [str(self.spot_properties['n_spots'])]
        # dstart_all = np.repeat(np.min(tt) + float(p_rot)/2.*np.arange(ntrials),
        #                        len(nspot_list))
        #
        #
        # # main loop for .in file writing of each time window & #spots
        # for n in range(len(dstart_all)):
        #     dstart = dstart_all[n]
        #     npts = np.sum((tt >= dstart) & (tt <= dstart+ddur))
        #     wndw = np.where((tt >= dstart) & (tt <= dstart+ddur)) # this could be combined w/ above...
        #
        #     # only run STSP on line if more than N epoch of data
        #     if npts >= 500:
        #
        #         window_dir = os.path.join(self.output_dir_path,
        #                                   "window{0:03d}".format(n))
        #         data_path = os.path.join(window_dir,
        #                                  "window{0:03d}.dat".format(n))
        #         if not os.path.exists(window_dir):
        #             os.mkdir(window_dir)
        #
        #         # write small chunk of LC to operate on
        #         dfn = open(data_path, 'w')
        #         for k in range(0,npts-1):
        #             dfn.write(str(tt[wndw[0][k]]) + ' ' +
        #                       str(ff[wndw[0][k]]) + ' ' +
        #                       str(ee[wndw[0][k]]) + '\n')
        #         dfn.close()


    @property
    def window_dirs(self):
        """Get all of the window??? directory paths, sort them"""
        return sorted(glob(os.path.join(self.output_dir_path, 'window*')))

    def create_runs(self):
        """
        Make run??? dirs in each window??? dir.
        """
        for window in self.window_dirs:
            for restart in range(self.n_restarts):
                new_dir_path = os.path.join(window, 'run{0:03d}'.format(restart))
                if not os.path.exists(new_dir_path):
                    os.mkdir(new_dir_path)
                    upper_level_light_curve_path = glob(os.path.join(window, '*.dat'))[0]
                    shutil.copy(upper_level_light_curve_path, new_dir_path)

                in_file_name = '_'.join(new_dir_path.split(os.sep)[-2:]) + '.in'

                if restart == 0:
                    lower_level_light_curve_path = glob(os.path.join(new_dir_path, '*.dat'))[0]
                    self.write_unseeded_in_file(os.path.join(new_dir_path, in_file_name),
                                                              lower_level_light_curve_path)
                else:
                    initialized_path = os.path.join(new_dir_path, 'initialized.txt')
                    if not os.path.exists(initialized_path):
                        previous_dir_path = os.path.join(window, 'run{0:03d}'.format(restart-1))
                        previous_in_file = glob(os.path.join(previous_dir_path, '*.in'))[0]
                        seed_finalparam_file = os.path.basename(previous_in_file.split('.in')[0] +
                                                               '_finalparam.txt')
                        #seed_finalparam_dir = os.path.dirname(previous_in_file).split(os.sep)[-1]
                        seed_finalparam_path = seed_finalparam_file
                        self.write_seeded_in_file(os.path.join(new_dir_path, in_file_name),
                                                  lower_level_light_curve_path,
                                                  seed_finalparam_path)


    def write_unseeded_in_file(self, output_path, light_curve_path):
        """
        Parameters
        ----------
        output_path : str
            Path to the .in file to write

        light_curve_path : str
            Path to input light curve
        """
        all_dicts = self.planet_properties
        for d in [self.stellar_properties, self.spot_properties,
                  self.action_properties]:
            all_dicts.update(d)

        light_curve = np.loadtxt(light_curve_path)
        all_dicts['action'] = 'M'
        all_dicts['lightcurve_path'] = light_curve_path.split(os.sep)[-1]
        all_dicts['start_fit_time'] = light_curve[0, 0]
        all_dicts['fit_duration_days'] = light_curve[-1, 0] - light_curve[0, 0]
        all_dicts['noise_corrected_max'] = np.max(gaussian_filter(light_curve[:, 1], 10))

        template = open('unseeded.in').read()
        in_file = template.format(**all_dicts)
        with open(output_path, 'w') as f:
            f.write(in_file)

            ## For spots with fixed latitudes at the equator:
            #for spot in range(all_dicts['n_spots']):
            #    f.write("1.57079632679\n")

    def write_seeded_in_file(self, output_path, light_curve_path,
                             seed_finalparam_path, spot_radius_sigma=0.01,
                             spot_angle_sigma=0.02):
        """
        Parameters
        ----------
        output_path : str
            Path to the .in file to write

        light_curve_path : str
            Path to input light curve
        """
        all_dicts = self.planet_properties
        for d in [self.stellar_properties, self.spot_properties,
                  self.action_properties]:
            all_dicts.update(d)

        light_curve = np.loadtxt(light_curve_path)
        # Seed options:
        all_dicts['action'] = 'T'
        all_dicts['spot_radius_sigma'] = spot_radius_sigma
        all_dicts['spot_angle_sigma'] = spot_angle_sigma
        all_dicts['seed_finalparam_path'] = seed_finalparam_path

        # Normal options:
        all_dicts['lightcurve_path'] = light_curve_path.split(os.sep)[-1]
        all_dicts['start_fit_time'] = light_curve[0, 0]
        all_dicts['fit_duration_days'] = light_curve[-1, 0] - light_curve[0, 0]
        all_dicts['noise_corrected_max'] = np.max(gaussian_filter(light_curve[:, 1], 10))

        template = open('seeded.in').read()
        in_file = template.format(**all_dicts)
        with open(output_path, 'w') as f:
            f.write(in_file)

            # For spots with fixed latitudes at the equator:
            # for spot in range(all_dicts['n_spots']):
            #     f.write("1.57079632679\n")
