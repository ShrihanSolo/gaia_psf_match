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

PSF_DATA_FILEPATH = "../../psf_data/kids_allfields.fits"
RESULTS_FILEPATH = "../results/"
CLUSTER_SUBSAMPLE_SIZE = 10000
NUMBER_OF_CLUSTERS = 40
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
des = read_kids_fits(PSF_DATA_FILEPATH, n = CLUSTER_SUBSAMPLE_SIZE)
print("Data read in.")

# Plot location of subsample of PSF stars
match.plot_sanity_test(des['coord'], fold = RESULTS_FILEPATH, BAND = BAND)

# Perform clustering on subsample of PSF stars
centroids = match.perform_clustering(des, NUMBER_OF_CLUSTERS, CLUSTER_SUBSAMPLE_SIZE)

# Save centroids array to int_data
np.save(INT_DATA_PATH + "kids_centroids.npy", centroids)
