import numpy as np
from astropy.stats import SigmaClip, sigma_clipped_stats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

plotpar = {'axes.labelsize': 25,
           'font.size': 22,
           'legend.fontsize': 20,
           'xtick.labelsize': 25,
           'ytick.labelsize': 25,
           'text.usetex': False}
plt.rcParams.update(plotpar)


def plot_original_image(img_F1000W_data, img_F1280W_data, img_F1800W_data,
                        path_output):

    mask = np.isfinite(img_F1000W_data)
    _, med_F1000W, sig_F1000W = sigma_clipped_stats(img_F1000W_data[mask],
                                                    sigma=5.0, maxiters=5)
    mask = np.isfinite(img_F1280W_data)
    _, med_F1280W, sig_F1280W = sigma_clipped_stats(img_F1280W_data[mask],
                                                    sigma=5.0, maxiters=5)
    mask = np.isfinite(img_F1800W_data)
    _, med_F1800W, sig_F1800W = sigma_clipped_stats(img_F1800W_data[mask],
                                                    sigma=5.0, maxiters=5)

    tlabel = 'Original image'
    xlabel = 'x [MIRIM pixel]'
    ylabel = 'y [MIRIM pixel]'
    blabel = 'MJy sr$^{-1}$'
    cmap = 'binary'

    vmin_array = [med - 1 * sig for med, sig in
                  zip([med_F1000W, med_F1280W, med_F1800W],
                      [sig_F1000W, sig_F1280W, sig_F1800W])]
    vmax_array = [med + 1 * sig for med, sig in
                  zip([med_F1000W, med_F1280W, med_F1800W],
                      [sig_F1000W, sig_F1280W, sig_F1800W])]

    fig, axs = plt.subplots(1, 3, figsize=(30, 15))
    axs = axs.ravel()

    data_array = [img_F1000W_data, img_F1280W_data, img_F1800W_data]
    filters_array = ['F1000W', 'F1280W', 'F1800W']

    for ax, img, vmin, vmax, filter_label in zip(axs, data_array, vmin_array,
                                                 vmax_array, filters_array):
        cax = ax.imshow(img, vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
        ax.text(100, 200, filter_label)
        ax_divider = make_axes_locatable(ax)
        cax1 = ax_divider.append_axes('right', size='3%', pad='2%')
        cb = fig.colorbar(cax, cax=cax1)
        cb.ax.set_ylabel(r'{0}'.format(blabel))
        ax.set_xlabel(r'{0}'.format(xlabel))
        ax.set_ylabel(r'{0}'.format(ylabel))
        ax.set_title(r'{0}'.format(tlabel))
    plt.tight_layout()

    plt.savefig(path_output + 'original_data.png', dpi=300,
                bbox_inches='tight')


def plot_skysub_image(img_F1000W_skysub, img_F1280W_skysub, img_F1800W_skysub,
                      centroids, path_output):
    mask = np.isfinite(img_F1000W_skysub)
    _, med_F1000W, sig_F1000W = sigma_clipped_stats(img_F1000W_skysub[mask],
                                                    sigma=5.0, maxiters=5)
    mask = np.isfinite(img_F1280W_skysub)
    _, med_F1280W, sig_F1280W = sigma_clipped_stats(img_F1280W_skysub[mask],
                                                    sigma=5.0, maxiters=5)
    mask = np.isfinite(img_F1800W_skysub)
    _, med_F1800W, sig_F1800W = sigma_clipped_stats(img_F1800W_skysub[mask],
                                                    sigma=5.0, maxiters=5)

    tlabel = 'Image sky subtracted'
    xlabel = 'x [MIRIM pixel]'
    ylabel = 'y [MIRIM pixel]'
    blabel = 'MJy sr$^{-1}$'
    cmap = 'binary'

    vmin_array = [med - .1 * sig for med, sig in
                  zip([med_F1000W, med_F1280W, med_F1800W],
                      [sig_F1000W, sig_F1280W, sig_F1800W])]
    vmax_array = [med + .1 * sig for med, sig in
                  zip([med_F1000W, med_F1280W, med_F1800W],
                      [sig_F1000W, sig_F1280W, sig_F1800W])]

    fig, axs = plt.subplots(1, 3, figsize=(30, 15))
    axs = axs.ravel()
    data_array = [img_F1000W_skysub, img_F1280W_skysub, img_F1800W_skysub]
    filter_array = ['F1000W', 'F1280W', 'F1800W']

    for ax, img, vmin, vmax, filter_label,xy_filter in zip(axs, data_array,
                                                           vmin_array,
                                                           vmax_array,
                                                           filter_array,
                                                           centroids):
        cax = ax.imshow(img, vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
        ax.scatter(xy_filter['xcentroid'], xy_filter['ycentroid'],
                   lw=0.5, s=15, marker='o', edgecolors='red',
                   facecolors='red')
        ax.scatter(xy_filter['xcentroid'], xy_filter['ycentroid'],
                   lw=0.5, s=15, marker='o', edgecolors='red',
                   facecolors='red')
        ax.text(100, 200, filter_label)
        ax_divider = make_axes_locatable(ax)
        cax1 = ax_divider.append_axes('right', size='3%', pad='2%')
        cb = fig.colorbar(cax, cax=cax1)
        cb.ax.set_ylabel(r'{0}'.format(blabel))
        ax.set_xlabel(r'{0}'.format(xlabel))
        ax.set_ylabel(r'{0}'.format(ylabel))
        ax.set_title(r'{0}'.format(tlabel))
    plt.tight_layout()
    plt.savefig(path_output + 'skysub_data.png', dpi=300, bbox_inches='tight')

    vmin_array = [med - .01 * sig for med, sig in
                  zip([med_F1000W, med_F1280W, med_F1800W],
                      [sig_F1000W, sig_F1280W, sig_F1800W])]
    vmax_array = [med + .3 * sig for med, sig in
                  zip([med_F1000W, med_F1280W, med_F1800W],
                      [sig_F1000W, sig_F1280W, sig_F1800W])]

    fig, axs = plt.subplots(1, 3, figsize=(20, 10), sharey=True, sharex=True)
    axs = axs.ravel()

    for ax, img, vmin, vmax, filter_label,xy_filter in zip(axs, data_array,
                                                           vmin_array,
                                                           vmax_array,
                                                           filter_array,
                                                           centroids):
        cax = ax.imshow(img, vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
        ax.text(780, 600, filter_label)
        ax_divider = make_axes_locatable(ax)
        ax.scatter(xy_filter['xcentroid'], xy_filter['ycentroid'],
                   lw=0.5, s=15, marker='o', edgecolors='red',
                   facecolors='red')
        ax.set_xlabel(r'{0}'.format(xlabel))
        ax.set_title(r'{0}'.format(tlabel))
        ax.set_xlim(xy_filter['xcentroid'] + 30, xy_filter['xcentroid'] - 30)
        ax.set_ylim(xy_filter['ycentroid'] + 30, xy_filter['ycentroid'] - 30)
    axs[0].set_ylabel(r'{0}'.format(ylabel))
    plt.tight_layout()
    plt.savefig(path_output + 'skysub_source.png', dpi=300,
                bbox_inches='tight')


def plot_choose_ap_radius(table_results_F1000W, table_results_F1280W,
                          table_results_F1800W,path_output):

    snr_F1000W = (table_results_F1000W['aperture_skysub_ujy'] /
                  table_results_F1000W['aperture_skysub_ujy_err'])
    snr_F1280W = (table_results_F1280W['aperture_skysub_ujy'] /
                  table_results_F1280W['aperture_skysub_ujy_err'])
    snr_F1800W = (table_results_F1800W['aperture_skysub_ujy'] /
                  table_results_F1800W['aperture_skysub_ujy_err'])

    plt.figure(figsize=(10, 7))

    plt.errorbar(table_results_F1000W['radius_ap'],snr_F1000W,
                 fmt='.-', label='F1000W', color='tab:blue')
    r_best_F1000W = table_results_F1000W['radius_ap'][np.nanargmax(snr_F1000W)]
    plt.axvline(x=r_best_F1000W, color='tab:blue')
    plt.text(r_best_F1000W, 170, f'r = {r_best_F1000W}', color='tab:blue')

    plt.errorbar(table_results_F1280W['radius_ap'],snr_F1280W,
                 fmt='.-', label='F1280W', color='tab:orange')
    r_best_F1280W = table_results_F1280W['radius_ap'][np.nanargmax(snr_F1280W)]
    plt.axvline(x=r_best_F1280W, color='tab:orange')
    plt.text(r_best_F1280W, 150, f'r = {r_best_F1280W}', color='tab:orange')

    plt.errorbar(table_results_F1800W['radius_ap'], snr_F1800W,
                 fmt='.-', label='F1800W', color='tab:green')
    r_best_F1800W = table_results_F1800W['radius_ap'][np.nanargmax(snr_F1800W)]
    plt.axvline(x=r_best_F1800W, color='tab:green')
    plt.text(r_best_F1800W, 130, f'r = {r_best_F1800W}', color='tab:green')

    plt.legend()
    plt.xlabel(r'Aperture radius')
    plt.ylabel('SNR Flux')
    plt.savefig(path_output + '_best_ap_radius.png', dpi=300,
                bbox_inches='tight')

    return r_best_F1000W, r_best_F1280W, r_best_F1800W


def plot_sed(results_F1000W, results_F1280W, results_F1800W,
             jmag, jmag_err, hmag, hmag_err, kmag, kmag_err, w1, w1_err,
             w2, w2_err, w3, w3_err, w4, w4_err, w1_cat, w1_cat_err,
             w2_cat, w2_cat_err, w3_cat, w3_cat_err, w4_cat, w4_cat_err,
             ch1, ch1_err, ch2, ch2_err, path_output):
    lam_miri = [18, 12.8, 10]
    FuJy_miri = [results_F1800W['aperture_skysub_ujy'][0],
                 results_F1280W['aperture_skysub_ujy'][0],
                 results_F1000W['aperture_skysub_ujy'][0]]
    e_FuJy_miri = [results_F1800W['aperture_skysub_ujy_err'][0],
                   results_F1280W['aperture_skysub_ujy_err'][0],
                   results_F1000W['aperture_skysub_ujy_err'][0]]

    J_lam = 1.235
    J_fzp = 1594
    J_fzp_err = 27.8
    H_lam = 1.662
    H_fzp = 1024
    H_fzp_err = 20
    K_lam = 2.159
    K_fzp = 666.7
    K_fzp_err = 12.6
    ch1_lam = 3.6
    ch1_fzp = 280.9
    ch1_fzp_err = 4.1
    ch2_lam = 4.5
    ch2_fzp = 179.7
    ch2_fzp_err = 2.6
    W1_lam = 3.3526
    W1_fzp = 306.682
    W1_fzp_err = 4.6
    W2_lam = 4.6028
    W2_fzp = 170.663
    W2_fzp_err = 2.6
    W3_lam = 11.5608
    W3_fzp = 29.045
    W3_fzp_err = 0.436
    W4_lam = 22.0883
    W4_fzp = 8.284
    W4_fzp_err = 0.124

    lam_2mass = np.array([J_lam, H_lam, K_lam])
    mag_2mass = np.array([jmag, hmag, kmag])
    mag_err_2mass = np.array([jmag_err, hmag_err, kmag_err])
    fzp_2mass = np.array([J_fzp, H_fzp, K_fzp])
    fzp_err_2mass = np.array([J_fzp_err, H_fzp_err, K_fzp_err])
    flux_2mass, flux_err_2mass = get_flux_from_mag(mag_2mass, mag_err_2mass,
                                                   fzp_2mass, fzp_err_2mass)

    lam_spitzer = np.array([ch1_lam, ch2_lam])
    mag_spitzer = np.array([ch1, ch2])
    mag_err_spitzer = np.array([ch1_err, ch2_err])
    fzp_spitzer = np.array([ch1_fzp, ch2_fzp])
    fzp_err_spitzer = np.array([ch1_fzp_err, ch2_fzp_err])
    flux_spitzer, flux_err_spitzer = get_flux_from_mag(mag_spitzer,
                                                       mag_err_spitzer,
                                                       fzp_spitzer,
                                                       fzp_err_spitzer)

    lam_wise = np.array([W1_lam, W2_lam, W3_lam, W4_lam])
    mag_wise = np.array([w1, w2, w3, w4])
    mag_err_wise = np.array([w1_err, w2_err, w3_err, w4_err])
    fzp_wise = np.array([W1_fzp, W2_fzp, W3_fzp, W4_fzp])
    fzp_err_wise = np.array([W1_fzp_err, W2_fzp_err, W3_fzp_err, W4_fzp_err])
    flux_wise, flux_err_wise = get_flux_from_mag(mag_wise, mag_err_wise,
                                                 fzp_wise, fzp_err_wise)

    mag_cat_wise = np.array([w1_cat, w2_cat, w3_cat, w4_cat])
    mag_err_cat_wise = np.array([w1_cat_err, w2_cat_err,
                                 w3_cat_err, w4_cat_err])
    flux_cat_wise, flux_cat_err_wise = get_flux_from_mag(mag_cat_wise,
                                                         mag_err_cat_wise,
                                                         fzp_wise,
                                                         fzp_err_wise)

    plt.figure(figsize=(10, 8))

    lam_all = np.concatenate((lam_miri,lam_spitzer,lam_wise,lam_wise,lam_2mass))
    flux_all = np.concatenate((FuJy_miri,flux_spitzer,flux_wise,
                               flux_cat_wise,flux_2mass))
    idx = np.argsort(lam_all)
    lam_all = lam_all[idx]
    flux_all = flux_all[idx]

    mask = ~np.isnan(lam_all+flux_all)
    plt.plot(lam_all[mask],flux_all[mask],'o-',color='k')
    plt.errorbar(lam_miri, FuJy_miri, yerr=e_FuJy_miri, fmt='o', label='MIRI')
    mask = ~np.isnan(flux_spitzer+flux_err_spitzer)
    if len(lam_spitzer[mask]>0):
        plt.errorbar(lam_spitzer[mask], flux_spitzer[mask],
                     yerr=flux_err_spitzer[mask], fmt='o', label='Spitzer')
    mask = ~np.isnan(flux_wise + flux_err_wise)
    if len(lam_wise[mask] > 0):
        plt.errorbar(lam_wise[mask], flux_wise[mask],
                     yerr=flux_err_wise[mask], fmt='o', label='WISE')
    mask = ~np.isnan(flux_cat_wise + flux_cat_err_wise)
    if len(lam_wise[mask] > 0):
        plt.errorbar(lam_wise[mask], flux_cat_wise[mask],
                     yerr=flux_cat_err_wise[mask], fmt='o', label='CatWISE')
    mask = ~np.isnan(flux_2mass + flux_err_2mass)
    if len(lam_2mass[mask] > 0):
        plt.errorbar(lam_2mass[mask], flux_2mass[mask],
                     yerr=flux_err_2mass[mask], fmt='o', label='2MASS')


    plt.xlabel(r'$\lambda~{\rm [\mu m]}$')
    plt.ylabel(r'F [${\rm \mu}$Jy]')
    plt.legend()
    plt.yscale('log')
    plt.savefig(path_output + '_sed.png', dpi=300, bbox_inches='tight')


def get_flux_from_mag(mag, mag_err, fzp, fzp_err):
    mask = ~np.isnan(mag+fzp)
    flux_2mass = np.ones(len(mag))*np.nan
    flux_2mass_err = np.ones(len(mag))*np.nan
    flux_2mass[mask] = fzp[mask]*10**(-0.4*mag[mask])
    mask = ~np.isnan(mag + mag_err + fzp + fzp_err)
    flux_2mass_err[mask] = np.sqrt((10**(-0.4*mag[mask]))**2*fzp_err[mask]**2 +
           (0.4*fzp[mask]*10**(-0.4*mag[mask])*np.log(10))**2*mag_err[mask]**2)

    return flux_2mass, flux_2mass_err
