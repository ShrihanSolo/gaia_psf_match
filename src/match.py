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


def plot_sanity_test(coord, n = 10000):
    plt.figure(figsize=(8,4.2))
    plt.subplot(111, projection="aitoff")
    plt.title("Sanity Test: PSF Stars in Surveys")
    plt.grid(True)
    
    # Choose n random stars
    coord = coord.sample(n=n)
    
    ra_rad = np.array([c.ra.wrap_at(180 * u.deg).radian for c in coord])
    dec_rad = np.array([c.dec.radian for c in coord])
    plt.scatter(ra_rad, dec_rad, s=0.1, alpha=0.8)
    plt.show()
    

def plot_cluster_test(ra_dec, centroids, cluster_num_array):
    """
    Plot the K-Means cluster test results, with clusters.
    Args:
        ra_dec: numpy array of ra and dec (2, n)
        centroids: numpy array of centroid positions 
        cluster_num_array: numpy array of cluster assignment, 
                            distance to centroid for each star 
    """
    
    # Sample 10000 random 2D points from numpy array
    idx = np.random.choice(ra_dec.shape[0], 10000, replace=False)
    rds = ra_dec[idx, :]
    
    # scatter plot of K-Means cluster
    plt.scatter(rds[:, 0],
                rds[:, 1],
                c=cluster_num_array[0][idx], s  = 0.1)
    
    # Centroid of the clusters
    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                marker='*',
                s=80,
                color='black')

    plt.title("Sanity Clustering Test")
    plt.xlabel("RA")
    plt.ylabel("Dec")


def sanity_separation_test(master_comb_df):
    plt.hist(master_comb_df["sep2d"], bins = np.logspace(-5, 3))
    plt.semilogx()
    plt.axvline(1, color = "k")
    plt.ylabel("Counts")
    plt.xlabel("Separation from Best Match (arcsec)")
    plt.title("Matched Stars: Separation in Arcsec")
    return

def sanity_crossmatch_test(master_comb_df, i = 0):
    """
    Plot a sanity check of the crossmatching process.
    Args:
        master_comb_df: pandas dataframe of matched stars
        i: index of the cluster to be plotted
    """
    
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))


    # Scatter plot for the first subplot
    ax[0].scatter(master_comb_df["cluster_ra"], master_comb_df["cluster_dec"], c = master_comb_df["mag0"], cmap = "cividis", 
                vmin = 18, vmax = 23, label = "HSC")
    ax[0].scatter(master_comb_df["gaia_ra"], master_comb_df["gaia_dec"], s=5, color="orange", label = "Gaia")
    ax[0].legend(loc = 'upper right')

    # Set labels and limits for the first subplot
    ax[0].set_xlabel("RA")
    ax[0].set_ylabel("Dec")
    ax[0].set_xlim(centroids[i][0] - 0.15, centroids[i][0] + 0.15)
    ax[0].set_ylim(centroids[i][1] - 0.15, centroids[i][1] + 0.15)

    ax[0].set_title("Pre-Matching, Sample Region")

    # Scatter plot for the second subplot
    match_idx = (master_comb_df["matched"] == 1)
    ax[1].scatter(master_comb_df["cluster_ra"][match_idx], master_comb_df["cluster_dec"][match_idx], c = master_comb_df["mag0"][match_idx], cmap = "cividis",
                vmin = 18, vmax = 23)
    ax[1].scatter(master_comb_df["gaia_ra"][match_idx], master_comb_df["gaia_dec"][match_idx], s=5, color="orange")

    # Set labels and limits for the second subplot
    ax[1].set_xlabel("RA")
    ax[1].set_ylabel("Dec")
    ax[1].set_xlim(centroids[i][0] - 0.15, centroids[i][0] + 0.15)
    ax[1].set_ylim(centroids[i][1] - 0.15, centroids[id][1] + 0.15)

    ax[1].set_title("Post-Matching, Sample Region")
    
    return fig, ax


def query_gaia_for_cluster(ra, dec, dist, lim = 1e6, verbose = False):
    """
    Query Gaia DR3 for upto 'lim' stars within a given radius 'dist' of a given center.
    Args:
        ra: center ra of the region
        dec: center dec of the region
        dist: radius of the region
        lim: limit of stars to be queried
        verbose: print the query
    """
    
    # Define the center coordinates of your region and the search radius
    center_coordinates = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree))
    search_radius = dist * u.deg  # Adjust the radius as needed
    
    Q = """
    select top {limit} ra, dec, phot_g_mean_mag, in_qso_candidates, in_galaxy_candidates, non_single_star, astrometric_excess_noise
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
    POINT({ra}, {dec}),
    CIRCLE(ra, dec, {radius}))
    order by source_id
    """

    Gaia.ROW_LIMIT = -1
    query = Q.format(ra=ra,dec=dec,radius=search_radius.value, limit = int(lim))
    if verbose:
        print(query)
    job = Gaia.launch_job(query, dump_to_file=True)
    result_table = job.get_results()
    gaia_table = result_table.to_pandas()
    
    # Alex DW Recommended Star-Galaxy Cut. Objects which pass this cut are stars, otherwise, galaxy.
    
    gaia_table["is_star"] = (np.log10(np.maximum(gaia_table['astrometric_excess_noise'], 1e-12)) < np.maximum((gaia_table['phot_g_mean_mag']-18.2)*.3+.2,.3))
    gaia_table.drop(columns=['astrometric_excess_noise', 'phot_g_mean_mag'], inplace=True)
    
    if len(gaia_table) == lim:
        print(f"Warning: Limit of {lim} reached. Increase limit or decrease radius (increase number of clusters).")
    
    return gaia_table


def match_cluster_to_gaia(cluster_num_array, ra_dec, cluster_info, cluster_num, MATCH_LIM = 1 * u.arcsec):
    """
    Match stars in cluster 'cluster_num' to gaia stars. Create a table with gaia coords, cluster coords, and flag for matched stars.
    Args:
        cluster_num_array: numpy array of cluster assignment, distance to centroid for each star 
        ra_dec: numpy array of ra and dec (2, n)
        cluster_info: pandas dataframe with cluster-specific info -> clusterno, centroids, max_dist.
        cluster_num: cluster number to be matched
        MATCH_LIM: maximum separation between a cluster star and a gaia star to be considered a match
    """
    
    # Query gaia and match stars to cluster    
    clust0_info = cluster_info.loc[cluster_num]
    print("R = {:.3f}".format(clust0_info["max_dist"]), end = ' | ')
    gaia0_tab = query_gaia_for_cluster(clust0_info["centroids"][0], clust0_info["centroids"][1], clust0_info["max_dist"])
    
    print("Queried.", end = ' ')
    
    cluster0 = SkyCoord(ra_dec[cluster_num_array[0] == cluster_num] * u.deg)
    mag0 = np.array(des['mag'][cluster_num_array[0] == cluster_num])
    gaia0 = SkyCoord(ra = gaia0_tab['ra'], dec = gaia0_tab['dec'], unit=u.deg)
    idx_clust, sep2d_clust, _ = cluster0.match_to_catalog_sky(gaia0)
    
    # Create table with gaia0 coords, cluster0 coords, and flag for matched stars
    comb_stars = pd.DataFrame({'matched': np.zeros(len(cluster0))})
    comb_stars.loc[sep2d_clust < MATCH_LIM] = 1
    comb_stars['sep2d'] = sep2d_clust.arcsec # in arcsec
    comb_stars['mag0'] = mag0
    comb_stars['is_star'] = np.array(gaia0_tab['is_star'].iloc[idx_clust])
    comb_stars['in_qso_candidates'] = np.array(gaia0_tab['in_qso_candidates'].iloc[idx_clust])
    comb_stars['in_galaxy_candidates'] = np.array(gaia0_tab['in_galaxy_candidates'].iloc[idx_clust])
    comb_stars['non_single_star'] = np.array(gaia0_tab['non_single_star'].iloc[idx_clust])

    get_ra = lambda x: x.ra.degree
    get_dec = lambda x: x.dec.degree

    comb_stars["gaia_ra"] = gaia0[idx_clust].ra.degree
    comb_stars["gaia_dec"] = gaia0[idx_clust].dec.degree
    comb_stars["cluster_ra"] = cluster0.ra.degree
    comb_stars["cluster_dec"] = cluster0.dec.degree
    
    return comb_stars


def plot_match_completeness(master_comb_df, bounds = [15, 21]):
    """
    Plot the magnitude distribution of the matched stars, and the matched stars that are also Gaia QSO and Galaxy candidates.
    Args:
        master_comb_df: Pandas dataframe of matched stars.
        bounds: bounds of the magnitude distribution to be plotted.
    """

    match_idx = (master_comb_df["matched"] == 1)
    super_match_idx = (master_comb_df[match_idx]["non_single_star"] == 0) & (master_comb_df[match_idx]["in_galaxy_candidates"] == False)
    plt.hist(master_comb_df["mag0"], bins = np.linspace(bounds[0], bounds[1], 40), label = "All PSF Stars", alpha = 0.5, color = "purple")
    plt.hist(master_comb_df["mag0"][~match_idx], bins = np.linspace(bounds[0], bounds[1], 40), label = "Without Gaia Match", alpha = 0.5, color = "red")
    plt.hist(master_comb_df["mag0"][match_idx][super_match_idx],bins = np.linspace(bounds[0], bounds[1], 40), label = "With Gaia Match", alpha = 0.5, color = "blue")
    plt.hist(master_comb_df["mag0"][match_idx][master_comb_df[match_idx]["in_qso_candidates"] == True], bins = np.linspace(bounds[0], bounds[1], 40), label = "Matched Gaia QSO Candidates", color = "black")
    plt.hist(master_comb_df["mag0"][match_idx][master_comb_df[match_idx]["is_star"] == False], bins = np.linspace(bounds[0], bounds[1], 40), label = "Failed Star-Galaxy Cut", color = "lime")
    plt.hist(master_comb_df["mag0"][match_idx][master_comb_df[match_idx]["in_galaxy_candidates"] == True], bins = np.linspace(bounds[0], bounds[1], 40), label = "Matched Gaia Galaxy Candidates",color = "green")
    plt.hist(master_comb_df["mag0"][match_idx][master_comb_df[match_idx]["non_single_star"] > 0], bins = np.linspace(bounds[0], bounds[1], 40), label = "Non-Single Star > 0",color = "grey")
    plt.title(f"DES Subsample of {len(master_comb_df)} Stars: Gaia Match Completeness")
    plt.xlabel("Z-band Magnitude")
    plt.ylabel("Number of Stars")
    plt.legend(fontsize = "x-small")
    return
    
def concatenate_int_data(fold):
    files = os.listdir(fold)
    df_list = []
    for file in files:
        df_list.append(pd.read_csv(fold + file, index_col=0))
        master_comb_df = pd.concat(df_list)
    return master_comb_df

def galaxy_ratio_plot(master_comb_df, bounds = [15, 21], w = 0.1):
    """
    Plot the ratio of Gaia galaxies to first: all matched PSFs, and second to all PSFs.
    Args:
        master_comb_df: Pandas dataframe of matched stars.
        bounds: bounds of the magnitude distribution to be plotted.
        w: width of the bars
    """
    # Define the bins
    bins = np.linspace(bounds[0], bounds[1], 40)
    all_match = np.histogram(master_comb_df["mag0"][match_idx], bins = bins)
    galaxy_match = np.histogram(master_comb_df["mag0"][match_idx][master_comb_df[match_idx]["in_galaxy_candidates"] == True], bins = bins)
    failed_star_galaxy_cut = np.histogram(master_comb_df["mag0"][match_idx][master_comb_df[match_idx]["is_star"] == False], bins = bins)

    mid = lambda x: x[:-1] + np.diff(x)/2

    plt.bar(mid(failed_star_galaxy_cut[1]), failed_star_galaxy_cut[0] / all_match[0], label = "Fraction Matched Failing Star-Galaxy Cut",color = "orange", width=0.14)
    plt.bar(mid(galaxy_match[1]), galaxy_match[0] / all_match[0], label = "Fraction Matched Gaia Galaxy Candidates",color = "green", width=0.14)
    plt.title("Ratio of Non-Star Matches to All Gaia Matched Objects")
    plt.xlabel("Z-Band Magnitude")
    plt.ylabel("Fraction of DESY3 Stars")
    plt.legend()
    plt.show()

    plt.bar(mid(failed_star_galaxy_cut[1]), failed_star_galaxy_cut[0] / all_match[0], label = "Fraction Matched Failing Star-Galaxy Cut",color = "orange", width=0.14)
    plt.bar(mid(galaxy_match[1]), galaxy_match[0] / all_match[0], label = "Fraction Matched Gaia Galaxy Candidates",color = "green", width=0.14)
    plt.title("Ratio of Gaia Non-Star Matches to All PSF Stars")
    plt.legend()
    plt.xlabel("Z-Band Magnitude")
    plt.ylabel("Fraction of DESY3 Stars")
    return 