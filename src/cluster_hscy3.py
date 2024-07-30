from astropy.io import fits
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import vq, kmeans
import scipy
import os
import h5py as h5

import match

# Choose a band for the clustering to be based on - 'g', 'r', 'i', 'z', 'y'
BAND = 'i'

PSF_DATA_FILEPATH = "../../psf_data/hscy3_allfields.h5"
RESULTS_FILEPATH = "../results/"
CLUSTER_SUBSAMPLE_SIZE = 10000
NUMBER_OF_CLUSTERS = 20
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

# Read in DES Data
des = read_hscy3_h5(PSF_DATA_FILEPATH, n = CLUSTER_SUBSAMPLE_SIZE)
print("Data read in.")

# Plot location of subsample of PSF stars
match.plot_sanity_test(des['coord'], fold = RESULTS_FILEPATH, BAND = BAND)

# Perform clustering on subsample of PSF stars
centroids = match.perform_clustering(des, NUMBER_OF_CLUSTERS, CLUSTER_SUBSAMPLE_SIZE)

# Save centroids array to int_data
np.save(INT_DATA_PATH + "hscy3_centroids.npy", centroids)
