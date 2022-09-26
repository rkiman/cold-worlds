
import coldworlds as cw
import numpy as np

object = ''
path = ''

path_F1000W = path + 'jw02124-o015_t002_miri_f1000w_i2d.fits'
pathF1280W = path + 'jw02124-o015_t002_miri_f1280w_i2d.fits'
pathF1800W = path + 'jw02124-o015_t002_miri_f1800w_i2d.fits'

centroid_F1000W = [753,626]
centroid_F1280W = [753,626]
centroid_F1800W = [753,626]

px_centroids = [centroid_F1000W, centroid_F1280W, centroid_F1800W]

jmag=np.nan
jmag_err=np.nan
hmag=np.nan
hmag_err=np.nan
kmag=np.nan
kmag_err=np.nan
w1=np.nan
w1_err=np.nan
w2=np.nan
w2_err=np.nan
w3=np.nan
w3_err=np.nan
w4=np.nan
w4_err=np.nan
w1_cat=np.nan
w1_cat_err=np.nan
w2_cat=np.nan
w2_cat_err=np.nan
w3_cat=np.nan
w3_cat_err=np.nan
w4_cat=np.nan
w4_cat_err=np.nan
ch1=np.nan
ch1_err=np.nan
ch2=np.nan
ch2_err=np.nan

cw.get_photometry(object, path_F1000W, pathF1280W, pathF1800W, path,
                  px_centroids, jmag, jmag_err, hmag, hmag_err, kmag, kmag_err,
                  w1, w1_err, w2, w2_err, w3, w3_err, w4, w4_err,
                  w1_cat, w1_cat_err, w2_cat, w2_cat_err, w3_cat, w3_cat_err,
                  w4_cat, w4_cat_err, ch1, ch1_err, ch2, ch2_err)