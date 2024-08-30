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
NUMBER_OF_CLUSTERS = 40
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

# Read in DES Data
des = read_kids_fits(PSF_DATA_FILEPATH, n = TOTAL_SUBSAMPLE_SIZE)
print("Data read in.")

# Save des subsample in int_data/des
des.to_csv(INT_DATA_PATH + "kids/" + "kids_" + str(BAND) + ".csv")
print("Subsample saved.")

# Load centroids array from int_data
centroids = np.load(INT_DATA_PATH + "kids_centroids.npy")

# Get assignments for all stars in DES
cluster_num_array, cluster_info = match.get_assignments(des, centroids)
print("DES Stars Assigned.")

# Save cluster_num_array and cluster_info in int_data
np.save(INT_DATA_PATH + "cluster_num_array_kids" + str(BAND) + ".npy", cluster_num_array)
cluster_info.to_csv(INT_DATA_PATH + "cluster_info.csv")

# Plot the clusters with color
ra_dec = np.array([des['ra'], des['dec']]).T
match.plot_cluster_test(ra_dec, centroids, cluster_num_array, fold = RESULTS_FILEPATH, BAND = BAND)

# Query Gaia for each cluster
# for cluster_num in range(centroids.shape[0]):
for cluster_num in range(centroids.shape[0]):
    clust0_info = cluster_info.loc[cluster_num]
    print("R = {:.3f}".format(clust0_info["max_dist"]), end = ' | ')
    gaia0_tab = match.query_gaia_for_cluster(clust0_info["centroids"][0], 
                                       clust0_info["centroids"][1], 
                                       clust0_info["max_dist"],
                                       verbose=False)
    print("Queried.", end = ' ')
    
    # Save the gaia table in int_data
    gaia0_tab.to_feather(INT_DATA_PATH + "gaia_kids/" + "gaia" + str(cluster_num) + ".feather")
    print("Saved.")