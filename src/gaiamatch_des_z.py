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
BAND = 'z'
PSF_DATA_FILEPATH = "../../psf_data/psf_y3a1-v29.fits"
RESULTS_FILEPATH = "../results/"
TOTAL_SUBSAMPLE_SIZE = 1000000
MATCH_LIM = 1 * u.arcsec
INT_DATA_PATH = "../../int_data/"

def read_des_fits(file_path, band, n = int(1e6)):
    """
    Read in the DES fits file and return a pandas dataframe with ra, dec, mag and band columns.
    Args: 
        file_path: path to the fits file
        band: band to be read in
        n: subsample of stars to be read in within specified band
    Returns:
        des: pandas dataframe with ra, dec, mag and band columns
    """
    
    # Read in the fits file and close it
    hdul = fits.open(file_path)
    
    # hdul[1].data is a numpy recarray. Get the ra, dec, mag and band columns   
    cols = ['ra', 'dec', 'mag', 'band']
    zidx = np.random.choice(np.where(hdul[1].data['band'] == band)[0], size = n, replace = False)
    data = {col: hdul[1].data[col][zidx] for col in cols}
    hdul.close()

    des = pd.DataFrame(data)
    

    # Combine ra and dec into a sky coord array
    des['coord'] = SkyCoord(ra=des['ra'], dec=des['dec'], unit = 'deg')
    return des

print("Starting DES Gaia Crossmatch for Band " + str(BAND) + ".")

# Alter results filepath to include band
RESULTS_FILEPATH_BAND = RESULTS_FILEPATH + str(BAND) + "band" + "/"
INT_DATA_PATH_BAND = INT_DATA_PATH + str(BAND) + "data" + "/"

# Read in DES Data
des = read_des_fits(PSF_DATA_FILEPATH, BAND, n = TOTAL_SUBSAMPLE_SIZE)
print("Data read in.")

# Load centroids array from int_data
centroids = np.load(INT_DATA_PATH + "centroids.npy")

# Get assignments for all stars in DES
cluster_num_array, cluster_info = match.get_assignments(des, centroids)
print("DES Stars Assigned.")

# Match Gaia for stars in the clusters 
for i in range(centroids.shape[0]):
    print("Cluster " + str(i) + ":", end = ' ')
    gaia0_tab = pd.read_feather(INT_DATA_PATH + "gaia/" + "gaia" + str(i) + ".feather")
    comb_clusteri = match.match_cluster_to_gaia(gaia0_tab, des, cluster_num_array, cluster_info, i)
    print("Matched.")
    comb_clusteri.to_csv(INT_DATA_PATH + str(BAND) + "data/" + "cluster_" + str(BAND) + "_" + str(i) + ".csv")
    

# Concatenate all the clusters
master_comb_df = match.concatenate_int_data(INT_DATA_PATH_BAND)
master_comb_df.to_csv(RESULTS_FILEPATH_BAND + "DES_MATCH_BAND" + str(BAND) + ".csv")
print("Concatenated Master DF.")

# Plot matching tests and results
match.sanity_separation_test(master_comb_df, fold = RESULTS_FILEPATH_BAND, BAND = BAND)
match.plot_match_completeness(master_comb_df, fold = RESULTS_FILEPATH_BAND, BAND = BAND)
match.galaxy_ratio_plot(master_comb_df, fold = RESULTS_FILEPATH_BAND, BAND = BAND)
print("Plotting Complete.")

