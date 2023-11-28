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

# Read in DES Data
des = read_des_fits(PSF_DATA_FILEPATH, BAND, n = TOTAL_SUBSAMPLE_SIZE)
print("Data read in.")

# Save des subsample in int_data/des
des.to_csv(INT_DATA_PATH + "des/" + "des_" + str(BAND) + ".csv")

# Load centroids array from int_data
centroids = np.load(INT_DATA_PATH + "centroids.npy")

# Get assignments for all stars in DES
cluster_num_array, cluster_info = match.get_assignments(des, centroids)

# Query Gaia for each cluster
for cluster_num in range(centroids.shape[0]):
    clust0_info = cluster_info.loc[cluster_num]
    print("R = {:.3f}".format(clust0_info["max_dist"]), end = ' | ')
    gaia0_tab = match.query_gaia_for_cluster(clust0_info["centroids"][0], 
                                       clust0_info["centroids"][1], 
                                       clust0_info["max_dist"],
                                       verbose=False)
    print("Queried.", end = ' ')
    
    # Save the gaia table in int_data
    gaia0_tab.to_feather(INT_DATA_PATH + "gaia/" + "gaia" + str(cluster_num) + ".feather")
    print("Saved.")