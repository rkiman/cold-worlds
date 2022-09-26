#import glob
import os
#import shutil
#import sys
#import random
#import urllib
#import zipfile
#from astropy.coordinates import match_coordinates_sky, SkyCoord
from astropy.io import fits, ascii
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.table import Table, Column, vstack
#import astropy.units as u
#from astropy.visualization import LogStretch, LinearStretch, PercentileInterval, ManualInterval
#from astropy.nddata import Cutout2D
#from jwst import datamodels, associations
#from jwst.datamodels import ImageModel, dqflags
#from matplotlib import style, pyplot as plt, rcParams
#from matplotlib.colors import LogNorm
#from matplotlib.pyplot import figure
#from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import photutils
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils import Background2D, MedianBackground, ModeEstimatorBackground, MMMBackground
#from scipy import stats
#from scipy.interpolate import CubicSpline
#import asdf

from .plots import plot_original_image, plot_skysub_image, plot_choose_ap_radius, plot_sed


def get_photometry(name, path_F1000W, pathF1280W, pathF1800W, path_output,
                   px_centroids, jmag=np.nan, jmag_err=np.nan,
                   hmag=np.nan, hmag_err=np.nan, kmag=np.nan, kmag_err=np.nan,
                   w1=np.nan, w1_err=np.nan, w2=np.nan, w2_err=np.nan,
                   w3=np.nan, w3_err=np.nan, w4=np.nan, w4_err=np.nan,
                   w1_cat=np.nan, w1_cat_err=np.nan, w2_cat=np.nan,
                   w2_cat_err=np.nan, w3_cat=np.nan, w3_cat_err=np.nan,
                   w4_cat=np.nan, w4_cat_err=np.nan,
                   ch1=np.nan, ch1_err=np.nan, ch2=np.nan, ch2_err=np.nan):

    if not os.path.exists(path_output + 'plots/'):
        os.mkdir(path_output + 'plots/')

    path_output = path_output + 'plots/' + name + '_'

    r_array_F1000W = [1.0960273, 1.3812923, 1.6667134, 2.061799, 2.5121834,
                      4.333602, 6.063827]
    apcorr_array_F1000W = [5.0224957, 3.3492174, 2.5130143, 2.0127609,
                           1.6798427, 1.4577353, 1.2943636]
    r_in_F1000W = 6.063827
    r_out_F1000W = 10.189244

    r_array_F1280W = [1.3472188, 1.7155845, 2.12405, 2.526218, 3.1832995,
                      5.6633143, 7.760389]
    apcorr_array_F1280W = [5.028173, 3.3536472, 2.5175314, 2.0158858,
                           1.6842289, 1.4701607, 1.3109299]
    r_in_F1280W = 7.760389
    r_out_F1280W = 11.883059

    r_array_F1800W = [1.8482778, 2.3633413, 2.8778927, 3.4549882, 4.2038965,
                      7.36965, 10.345783]
    apcorr_array_F1800W = [5.039427, 3.362004, 2.5239372, 2.0221124, 1.68946,
                           1.4812119, 1.3316054]
    r_in_F1800W = 10.345783
    r_out_F1800W = 14.703575

    img_F1000W = fits.open(path_F1000W)
    img_F1280W = fits.open(pathF1280W)
    img_F1800W = fits.open(pathF1800W)

    img_F1000W_data = img_F1000W[1].data
    img_F1000W_data_err = img_F1000W[2].data
    filter_img_F1000W = 'F1000W'
    fwhm_F1000W = 0.32

    img_F1280W_data = img_F1280W[1].data
    img_F1280W_data_err = img_F1280W[2].data
    filter_img_F1280W = 'F1280W'
    fwhm_F1280W = 0.41

    img_F1800W_data = img_F1800W[1].data
    img_F1800W_data_err = img_F1800W[2].data
    filter_img_F1800W = 'F1800W'
    fwhm_F1800W = 0.58

    fwhm = [fwhm_F1000W, fwhm_F1280W, fwhm_F1800W]
    centroid_F1000W, centroid_F1280W, centroid_F1800W = px_centroids

    plot_original_image(img_F1000W_data, img_F1280W_data, img_F1800W_data,
                        path_output)

    r, sky_array = subtract_background(img_F1000W_data, img_F1280W_data,
                                       img_F1800W_data)
    img_F1000W_skysub, img_F1280W_skysub, img_F1800W_skysub = r

    xy_centroids = get_centroid(img_F1000W_skysub, img_F1280W_skysub,
                                img_F1800W_skysub, sky_array, fwhm,
                                centroid_F1000W, centroid_F1280W,
                                centroid_F1800W)

    xy_F1000W_tmp, xy_F1280W_tmp, xy_F1800W_tmp = xy_centroids

    plot_skysub_image(img_F1000W_skysub, img_F1280W_skysub, img_F1800W_skysub,
                      xy_centroids, path_output)

    i = 0
    for r, apcorr in zip(r_array_F1000W, apcorr_array_F1000W):
        results_F1000W = calculate_photometry(img_F1000W_data,
                                              img_F1000W_data_err,
                                              xy_F1000W_tmp,
                                              r=r,
                                              r_in=r_in_F1000W,
                                              r_out=r_out_F1000W,
                                              apcorr=apcorr)
        if i == 0:
            col_names = ['radius_ap', 'ap_corr']
            for x in results_F1000W.columns:
                col_names.append(x)
            table_results_F1000W = Table(names=col_names)
        row = [r, apcorr]
        for x in results_F1000W[0]:
            row.append(x)
        table_results_F1000W.add_row(row)
        i += 1

    i = 0
    for r, apcorr in zip(r_array_F1280W, apcorr_array_F1280W):
        results_F1280W = calculate_photometry(img_F1280W_data,
                                              img_F1280W_data_err,
                                              xy_F1280W_tmp,
                                              r=r,
                                              r_in=r_in_F1280W,
                                              r_out=r_out_F1280W,
                                              apcorr=apcorr)
        if i == 0:
            col_names = ['radius_ap', 'ap_corr']
            for x in results_F1280W.columns:
                col_names.append(x)
            table_results_F1280W = Table(names=col_names)
        row = [r, apcorr]
        for x in results_F1280W[0]:
            row.append(x)
        table_results_F1280W.add_row(row)
        i += 1

        i = 0
        for r, apcorr in zip(r_array_F1800W, apcorr_array_F1800W):
            results_F1800W = calculate_photometry(img_F1800W_data,
                                                  img_F1800W_data_err,
                                                  xy_F1800W_tmp,
                                                  r=r,
                                                  r_in=r_in_F1800W,
                                                  r_out=r_out_F1800W,
                                                  apcorr=apcorr)
            if i == 0:
                col_names = ['radius_ap', 'ap_corr']
                for x in results_F1800W.columns:
                    col_names.append(x)
                table_results_F1800W = Table(names=col_names)
            row = [r, apcorr]
            for x in results_F1800W[0]:
                row.append(x)
            table_results_F1800W.add_row(row)
            i += 1

    radii = plot_choose_ap_radius(table_results_F1000W, table_results_F1280W,
                                  table_results_F1800W, path_output)

    r_best_F1000W = radii[0]
    r_best_F1280W = radii[1]
    r_best_F1800W = radii[2]

    results_F1000W = table_results_F1000W[table_results_F1000W['radius_ap'] == r_best_F1000W]
    results_F1280W = table_results_F1280W[table_results_F1280W['radius_ap'] == r_best_F1280W]
    results_F1800W = table_results_F1800W[table_results_F1800W['radius_ap'] == r_best_F1800W]

    plot_sed(results_F1000W, results_F1280W, results_F1800W,
             jmag, jmag_err, hmag, hmag_err, kmag, kmag_err, w1, w1_err,
             w2, w2_err, w3, w3_err, w4, w4_err, w1_cat, w1_cat_err,
             w2_cat, w2_cat_err, w3_cat, w3_cat_err, w4_cat, w4_cat_err,
             ch1, ch1_err, ch2, ch2_err, path_output)

    final_table = vstack([results_F1000W, results_F1280W, results_F1800W])
    final_table['filter'] = ['F1000W','F1280W','F1800W']
    final_table.write(path_output + 'phot.csv',format='csv',overwrite=True)

def subtract_background(img_F1000W_data, img_F1280W_data, img_F1800W_data):

    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    # This is the background estimator
    mmm_bkg = MMMBackground()

    # F1000W filter
    # Mask all nan or inf pixels
    mask = np.full(np.shape(img_F1000W_data), False, dtype=bool)
    mask[np.isnan(img_F1000W_data)] = True
    mask[~np.isfinite(img_F1000W_data)] = True
    # Compute sky background
    sky_F1000W = Background2D(img_F1000W_data, box_size=(50, 50),
                              filter_size=(31, 31),
                              sigma_clip=sigma_clip, bkg_estimator=mmm_bkg,
                              coverage_mask=mask, fill_value=0.0)
    img_F1000W_skysub = img_F1000W_data - sky_F1000W.background

    # F1280W filter
    # Mask all nan or inf pixels
    mask = np.full(np.shape(img_F1280W_data), False, dtype=bool)
    mask[np.isnan(img_F1280W_data)] = True
    mask[~np.isfinite(img_F1280W_data)] = True
    # Compute sky background
    sky_F1280W = Background2D(img_F1280W_data, box_size=(50, 50),
                              filter_size=(31, 31),
                              sigma_clip=sigma_clip, bkg_estimator=mmm_bkg,
                              coverage_mask=mask, fill_value=0.0)
    img_F1280W_skysub = img_F1280W_data - sky_F1280W.background

    # F1800W filter
    # Mask all nan or inf pixels
    mask = np.full(np.shape(img_F1800W_data), False, dtype=bool)
    mask[np.isnan(img_F1800W_data)] = True
    mask[~np.isfinite(img_F1800W_data)] = True
    # Compute sky background
    sky_F1800W = Background2D(img_F1800W_data, box_size=(50, 50),
                              filter_size=(31, 31),
                              sigma_clip=sigma_clip, bkg_estimator=mmm_bkg,
                              coverage_mask=mask, fill_value=0.0)
    img_F1800W_skysub = img_F1800W_data - sky_F1800W.background

    sky_array = [sky_F1000W, sky_F1280W, sky_F1800W]

    return [img_F1000W_skysub, img_F1280W_skysub, img_F1800W_skysub], sky_array


def get_centroid(img_F1000W_skysub, img_F1280W_skysub, img_F1800W_skysub,
                 sky_array, fwhm, centroid_F1000W, centroid_F1280W,
                 centroid_F1800W):
    sky_F1000W, sky_F1280W, sky_F1800W = sky_array
    fwhm_F1000W, fwhm_F1280W, fwhm_F1800W = fwhm
    # 5 times the background rms
    threshold_F1000W = 5.0 * sky_F1000W.background_rms_median
    # Create DAOStarFinder instance
    dsf_F1000W = photutils.DAOStarFinder(threshold=threshold_F1000W,
                                         fwhm=fwhm_F1000W, exclude_border=True)
    # Run DAOStarFinder on the subtracted image and save the output in a table
    xy_F1000W_tmp = dsf_F1000W(img_F1000W_skysub)

    xy_F1000W_tmp = xy_F1000W_tmp[
        np.isclose(xy_F1000W_tmp['xcentroid'], centroid_F1000W[0], rtol=1e-2) *
        np.isclose(xy_F1000W_tmp['ycentroid'], centroid_F1000W[1], rtol=1e-2)]
    try:
        assert len(xy_F1000W_tmp) == 1
    except AssertionError as msg:
        print('Check centroid F1000W, there is more than one object close.')

    # 5 times the background rms
    threshold_F1280W = 5.0 * sky_F1280W.background_rms_median
    # Create DAOStarFinder instance
    dsf_F1280W = photutils.DAOStarFinder(threshold=threshold_F1280W,
                                         fwhm=fwhm_F1280W, exclude_border=True)
    # Run DAOStarFinder on the subtracted image and save the output in a table
    xy_F1280W_tmp = dsf_F1280W(img_F1280W_skysub)
    xy_F1280W_tmp = xy_F1280W_tmp[
        np.isclose(xy_F1280W_tmp['xcentroid'], centroid_F1280W[0], rtol=1e-2) *
        np.isclose(xy_F1280W_tmp['ycentroid'], centroid_F1280W[1], rtol=1e-2)]
    try:
        assert len(xy_F1280W_tmp) == 1
    except AssertionError as msg:
        print('Check centroid F1280W, there is more than one object close.')

    # 5 times the background rms
    threshold_F1800W = 5.0 * sky_F1800W.background_rms_median
    # Create DAOStarFinder instance
    dsf_F1800W = photutils.DAOStarFinder(threshold=threshold_F1800W,
                                         fwhm=fwhm_F1800W, exclude_border=True)
    # Run DAOStarFinder on the subtracted image and save the output in a table
    xy_F1800W_tmp = dsf_F1800W(img_F1800W_skysub)
    xy_F1800W_tmp = xy_F1800W_tmp[
        np.isclose(xy_F1800W_tmp['xcentroid'], centroid_F1800W[0], rtol=1e-2) *
        np.isclose(xy_F1800W_tmp['ycentroid'], centroid_F1800W[1], rtol=1e-2)]
    try:
        assert len(xy_F1800W_tmp) == 1
    except AssertionError as msg:
        print('Check centroid F1800W, there is more than one object close.')

    return xy_F1000W_tmp, xy_F1280W_tmp, xy_F1800W_tmp


def calculate_photometry(img_data, img_data_err, xy_tmp, r, r_in, r_out,
                         apcorr):
    # Define the positions
    positions = np.stack((xy_tmp['xcentroid'], xy_tmp['ycentroid']), axis=-1)

    # print(r'Aperture radii used:')
    # print(r' r0 = {0:.3f} MIRIM pixel'.format(r))

    # Define the circular apertures
    circular_aperture_r = CircularAperture(positions, r=r)

    # Run the aperture photometry
    phot_tmp = aperture_photometry(img_data, circular_aperture_r,
                                   error=img_data_err, method='exact')

    # Define the annulus aperture
    annulus_aperture = CircularAnnulus(positions, r_in, r_out)

    # Define the mask with only pixels in each annulus
    annulus_mask = annulus_aperture.to_mask(method='center')

    # The local sky for each star will be stored in this list
    local_sky_median = []
    local_sky_std = []

    # For each source
    for mask in annulus_mask:
        # Multiply the pixel values by the mask. Since the mask is either
        # 0 or 1, the only non-zero pixels are those in the circular annulus
        annulus_data = mask.multiply(img_data)

        # Keep only non-masked pixels with finite values
        ok = np.logical_and(mask.data > 0, np.isfinite(annulus_data))

        # If there are not at least 10 usable pixels in the annulus to compute
        # the local median sky, flag the star and remove it from the final
        # list later

        if np.sum(ok) >= 10:
            # From 2D to 1D array
            annulus_data_1d = annulus_data[ok]
            # Sigma-clipped median
            r = sigma_clipped_stats(annulus_data_1d, sigma=3.5, maxiters=5)
            _, median_sigclip, stdev_sigclip = r
        else:
            # Flagged value
            median_sigclip = -99.99
            stdev_sigclip = -99.99

        local_sky_median.append(median_sigclip)
        local_sky_std.append(stdev_sigclip)

    # Convert list into array
    local_sky_median = np.array(local_sky_median)
    local_sky_std = np.array(local_sky_std)

    phot_table = Table()

    # Useful info from DAOStarFinder
    phot_table['x'] = xy_tmp['xcentroid']
    phot_table['y'] = xy_tmp['ycentroid']
    phot_table['sharpness'] = xy_tmp['sharpness']
    phot_table['roundness1'] = xy_tmp['roundness1']
    phot_table['roundness2'] = xy_tmp['roundness2']
    phot_table['aperture_sum'] = phot_tmp['aperture_sum']
    phot_table['aperture_sum_err'] = phot_tmp['aperture_sum_err']

    # Save an ID
    phot_table['id'] = xy_tmp['id']

    # Local median sky
    phot_table['annulus_median'] = local_sky_median
    phot_table['annulus_std'] = local_sky_std

    # Aperture photometry
    local_sky = local_sky_median * circular_aperture_r[0].area
    local_sky_err = np.sqrt(circular_aperture_r[0].area ** 2
                            * local_sky_std ** 2 / annulus_aperture[0].area)
    aperture_skysub = phot_tmp['aperture_sum'] - local_sky  # MJy/sr
    aperture_skysub_err = np.sqrt(phot_tmp['aperture_sum_err'] ** 2
                                  + local_sky_err ** 2)

    phot_table['local_sky'] = local_sky
    phot_table['local_sky_err'] = local_sky_err
    phot_table['aperture_skysub'] = aperture_skysub
    phot_table['aperture_skysub_err'] = aperture_skysub_err

    omega = 0.0121 / 206265 ** 2
    aperture_skysub_jy = phot_table['aperture_skysub'] * omega * 1e6 * apcorr
    phot_table['aperture_skysub_ujy'] = aperture_skysub_jy * 1e6
    phot_table['aperture_skysub_ujy_err'] = (phot_table['aperture_skysub_err']
                                             * omega * 1e6 * apcorr * 1e6)

    return phot_table
