from astropy.io import fits
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import vq, kmeans
import scipy
from astroquery.gaia import Gaia
import h5py as h5
import os

import match

# Global Parameters
BAND = 'r'
PSF_DATA_FILEPATH = "../../psf_data/kids_allfields.fits"
RESULTS_FILEPATH = "../results/"
TOTAL_SUBSAMPLE_SIZE = 1000000
MATCH_LIM = 1 * u.arcsec
INT_DATA_PATH = "../../int_data/"

def read_kids_fits(file_path, n = int(1e6)):
    """
    Read a kids file and return the data as a pandas dataframe.
    Args:
        file_path: path to the hfits5 file
    Returns:
        data: pandas dataframe with the data from the fits file
    """
    kids_tab = fits.open(PSF_DATA_FILEPATH)[2].data
    idx = np.random.choice(np.arange(len(kids_tab)), size = n, replace = False)
    mag_r = np.array(kids_tab['STAR_MAG'])[idx]
    ra = np.array(kids_tab['ALPHA_J2000'])[idx]
    dec = np.array(kids_tab['DELTA_J2000'])[idx]
    data = {'ra': ra, 'dec': dec, 'mag': mag_r}
    data['coord'] = SkyCoord(ra=data['ra'], dec=data['dec'], unit = 'deg')
    kids_data = pd.DataFrame(data)
    
    return kids_data

print("Starting DES Gaia Crossmatch for Band " + str(BAND) + ".")

# Alter results filepath to include band
RESULTS_FILEPATH_BAND = RESULTS_FILEPATH + str(BAND) + "band_kids" + "/"
INT_DATA_PATH_BAND = INT_DATA_PATH + str(BAND) + "data_kids" + "/"

# Read in DES Data
des = read_kids_fits(PSF_DATA_FILEPATH, n = TOTAL_SUBSAMPLE_SIZE)
print("Data read in.")

# Load centroids array from int_data
centroids = np.load(INT_DATA_PATH + "kids_centroids.npy")

# Get assignments for all stars in DES
cluster_num_array, cluster_info = match.get_assignments(des, centroids)
print("DES Stars Assigned.")

# Match Gaia for stars in the clusters 
for i in range(centroids.shape[0]):
    print("Cluster " + str(i) + ":", end = ' ')
    gaia0_tab = pd.read_feather(INT_DATA_PATH + "gaia_kids/" + "gaia" + str(i) + ".feather")
    comb_clusteri = match.match_cluster_to_gaia(gaia0_tab, des, cluster_num_array, cluster_info, i)
    print("Matched.")
    comb_clusteri.to_csv(INT_DATA_PATH + str(BAND) + "data_kids/" + "cluster_" + str(BAND) + "_" + str(i) + ".csv")
    

# Concatenate all the clusters
master_comb_df = match.concatenate_int_data(INT_DATA_PATH_BAND)
master_comb_df.to_csv(RESULTS_FILEPATH_BAND + "KIDS_MATCH_BAND" + str(BAND) + ".csv")
print("Concatenated Master DF.")

# Plot matching tests and results
match.sanity_separation_test(master_comb_df, fold = RESULTS_FILEPATH_BAND, BAND = BAND)
match.plot_match_completeness(master_comb_df, fold = RESULTS_FILEPATH_BAND, BAND = BAND)
match.galaxy_ratio_plot(master_comb_df, fold = RESULTS_FILEPATH_BAND, BAND = BAND)
print("Plotting Complete.")

