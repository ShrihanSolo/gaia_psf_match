from astropy.io import fits
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import vq, kmeans
import scipy
from astroquery.gaia import Gaia
import os

import match

# Global Parameters
BAND = 'i'
PSF_DATA_FILEPATH = "../../psf_data/hscy3_allfields.h5"
RESULTS_FILEPATH = "../results/"
TOTAL_SUBSAMPLE_SIZE = 1000000
MATCH_LIM = 1 * u.arcsec
INT_DATA_PATH = "../../int_data/"

def read_hscy3_h5(file_path, n = int(1e6)):
    """
    Read an h5 file and return the data as a pandas dataframe.
    Args:
        file_path: path to the h5 file
    Returns:
        data: pandas dataframe with the data from the h5 file
    """
    with h5.File(file_path, 'r') as f:
        # Read specific fields
        idx = np.random.choice(np.arange(len(f['stars/mag_i'])), size = n, replace = False)
        mag_i = np.array(f['stars/mag_i'])[idx]
        ra = np.array(f['stars/ra'])[idx]
        dec = np.array(f['stars/dec'])[idx]
        data = {'ra': ra, 'dec': dec, 'mag': mag_i}
        data['coord'] = SkyCoord(ra=data['ra'], dec=data['dec'], unit = 'deg')
    
    return pd.DataFrame(data)

print("Starting DES Gaia Crossmatch for Band " + str(BAND) + ".")

# Alter results filepath to include band
RESULTS_FILEPATH_BAND = RESULTS_FILEPATH + str(BAND) + "band_hscy3" + "/"
INT_DATA_PATH_BAND = INT_DATA_PATH + str(BAND) + "data_hscy3" + "/"

# Read in DES Data
des = read_hscy3_h5(PSF_DATA_FILEPATH, n = TOTAL_SUBSAMPLE_SIZE)
print("Data read in.")

# Load centroids array from int_data
centroids = np.load(INT_DATA_PATH + "centroids.npy")

# Get assignments for all stars in DES
cluster_num_array, cluster_info = match.get_assignments(des, centroids)
print("DES Stars Assigned.")

# Match Gaia for stars in the clusters 
for i in range(centroids.shape[0]):
    print("Cluster " + str(i) + ":", end = ' ')
    gaia0_tab = pd.read_feather(INT_DATA_PATH + "gaia_hscy3/" + "gaia_hscy3_" + str(i) + ".feather")
    comb_clusteri = match.match_cluster_to_gaia(gaia0_tab, des, cluster_num_array, cluster_info, i)
    print("Matched.")
    comb_clusteri.to_csv(INT_DATA_PATH + str(BAND) + "data_hscy3/" + "cluster_" + str(BAND) + "_" + str(i) + ".csv")
    

# Concatenate all the clusters
master_comb_df = match.concatenate_int_data(INT_DATA_PATH_BAND)
master_comb_df.to_csv(RESULTS_FILEPATH_BAND + "DES_MATCH_BAND" + str(BAND) + ".csv")
print("Concatenated Master DF.")

# Plot matching tests and results
match.sanity_separation_test(master_comb_df, fold = RESULTS_FILEPATH_BAND, BAND = BAND)
match.plot_match_completeness(master_comb_df, fold = RESULTS_FILEPATH_BAND, BAND = BAND)
match.galaxy_ratio_plot(master_comb_df, fold = RESULTS_FILEPATH_BAND, BAND = BAND)
print("Plotting Complete.")

