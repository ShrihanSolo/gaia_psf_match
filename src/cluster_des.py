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

# Choose a band for the clustering to be based on - 'g', 'r', 'i', 'z', 'y'
BAND = 'i'

PSF_DATA_FILEPATH = "../../psf_data/psf_y3a1-v29.fits"
RESULTS_FILEPATH = "../results/iband/"
TOTAL_SUBSAMPLE_SIZE = 10000
CLUSTER_SUBSAMPLE_SIZE = 1000
NUMBER_OF_CLUSTERS = 200
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

# Plot location of subsample of PSF stars
match.plot_sanity_test(des['coord'], fold = RESULTS_FILEPATH, BAND = BAND)

# Perform clustering on subsample of PSF stars
centroids = match.perform_clustering(des, NUMBER_OF_CLUSTERS, CLUSTER_SUBSAMPLE_SIZE)

# Save centroids array to int_data
np.save(INT_DATA_PATH + "centroids.npy", centroids)
