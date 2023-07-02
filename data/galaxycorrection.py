import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import yaml
import argparse

import os
import sys
sys.path.append("/Users/akiratokiwa/Git/HSCSextansPMMeasurement")
from utils.utils import Zs, apply_async_pool, grid, cali_star_galerr

def initialize_axes(fig, pos, title):
    ax = fig.add_axes(pos)
    ax.set_title(title, fontsize=16)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return ax

def draw_color_map(ax, xedges, yedges, Z, norms, cmap='RdYlBu'):
    return ax.pcolormesh(xedges, yedges, Z, norm = norms, cmap=cmap)

def plot_corrmap(Zsg, Zsg_cl, Zra_g, Zdec_g, img_path, xedges, yedges):
    norms=colors.Normalize(vmin=-10, vmax=10)
    fig = plt.figure(figsize=(14,5))

    ax00 = initialize_axes(fig, [0, 0.5, 0.3, 0.5], "Relative Proper Motion(Before Correction)")
    ax01 = initialize_axes(fig, [0.3, 0.5, 0.3, 0.5], "Distortion Correction Map")
    ax02 = initialize_axes(fig, [0.6, 0.5, 0.3, 0.5], "Corrected Proper Motion")
    ax10 = initialize_axes(fig, [0, 0, 0.3, 0.5], "")
    ax11 = initialize_axes(fig, [0.3, 0, 0.3, 0.5], "")
    ax12 = initialize_axes(fig, [0.6, 0, 0.3, 0.5], "")
    ax3 = fig.add_axes([0.92, 0, 0.02, 1])

    mappable0 = draw_color_map(ax00, xedges, yedges, Zsg[1], norms)
    mappable1 = draw_color_map(ax01, xedges, yedges, Zra_g, norms)
    mappable2 = draw_color_map(ax02, xedges, yedges, Zsg_cl[1], norms)
    mappable3 = draw_color_map(ax10, xedges, yedges, Zsg[2], norms)
    mappable4 = draw_color_map(ax11, xedges, yedges, Zdec_g, norms)
    mappable5 = draw_color_map(ax12, xedges, yedges, Zsg_cl[2], norms)

    ax00.set_ylabel("Dec [deg]",fontsize=16)
    ax10.set_ylabel("Dec [deg]",fontsize=16)

    for ax in [ax10, ax11, ax12]:
        ax.set_xlabel("RA [deg]",fontsize=16)

    ax00.text(150, 1.3, r"$\mu_{\alpha}$", size=24,
            horizontalalignment="left",
            verticalalignment="top")

    ax10.text(150, 1.3, r"$\mu_{\delta}$", size=24,
            horizontalalignment="left",
            verticalalignment="top")

    fig.colorbar(mappable0,
                aspect=40, shrink=0.6,
                orientation='vertical', extend='both',cax=ax3)

    fig.savefig(img_path, bbox_inches='tight', pad_inches=0.05)


def main(star_path, gal_path, attenuation_path, output_dir, img_path, config_path):
    ms=pd.read_csv(star_path)
    mg=pd.read_csv(gal_path)
    m_ext=pd.read_csv(attenuation_path)
    with open(config_path, 'r') as stream:
        config_dict = yaml.safe_load(stream)
    xedges=np.arange(config_dict["data"]["xmin"], config_dict["data"]["xmax"], config_dict["process"]["gridtick"])
    yedges=np.arange(config_dict["data"]["ymin"], config_dict["data"]["ymax"], config_dict["process"]["gridtick"])
    print(len(mg), len(ms))

    obs_date1 = 56980
    # separate the data into two groups
    ms1 = ms[ms.mjd_x == obs_date1].copy()
    mg1 = mg[mg.mjd_x == obs_date1].copy()
    ms2 = ms[ms.mjd_x != obs_date1].copy()
    mg2 = mg[mg.mjd_x != obs_date1].copy()

    data_mg1 = apply_async_pool(8, grid,yedges, mg1, xedges, yedges)
    Znum_g1, Zra_g1, Zdec_g1, Zrasem_g1, Zdecsem_g1 = Zs(data_mg1, "pmra", "pmdec")
    data_mg2 = apply_async_pool(8, grid,yedges, mg2, xedges, yedges)
    Znum_g2, Zra_g2, Zdec_g2, Zrasem_g2, Zdecsem_g2 = Zs(data_mg2, "pmra", "pmdec")

    # Save the data
    for name, df in [('Znum_g1', Znum_g1), ('Zra_g1', Zra_g1), ('Zdec_g1', Zdec_g1), ('Zrasem_g1', Zrasem_g1), ('Zdecsem_g1', Zdecsem_g1),
                    ('Znum_g2', Znum_g2), ('Zra_g2', Zra_g2), ('Zdec_g2', Zdec_g2), ('Zrasem_g2', Zrasem_g2), ('Zdecsem_g2', Zdecsem_g2)]:
            df.to_csv(f'{output_dir}{name}.csv', index=False)
    
    ms1_tmp=pd.concat(apply_async_pool(8, cali_star_galerr, yedges,
                                    ms1, xedges, yedges, Zra_g1, Zdec_g1, Zrasem_g1, Zdecsem_g1))
    ms2_tmp=pd.concat(apply_async_pool(8, cali_star_galerr,yedges,
                                        ms2, xedges, yedges, Zra_g2, Zdec_g2, Zrasem_g2, Zdecsem_g2))

    # Concatenate and clean the dataframes
    msall = (
        pd.concat([ms1_tmp, ms2_tmp])
        .dropna(subset=["pmra_cl", "pmdec_cl", "pmra_galerr", "pmdec_galerr", "i_psfflux_mag", "gi"])
        .merge(m_ext, on="# object_id")
    )

    # Define new columns
    new_columns = {
        "gc": msall.g_psfflux_mag - msall.a_g,
        "rc": msall.r_psfflux_mag - msall.a_r,
        "ic": msall.i_psfflux_mag - msall.a_i,
        "draerr": np.sqrt(msall.i_sdsscentroid_raerr**2 + (msall.i_ra_err*0.168)**2)*1000,
        "ddecerr": np.sqrt(msall.i_sdsscentroid_decerr**2 + (msall.i_dec_err*0.168)**2)*1000
    }
    # Add the new columns
    msall = msall.assign(**new_columns)

    new_columns2 = {
        "gr": msall.gc - msall.rc,
        "ri": msall.rc - msall.ic,
        "gi": msall.gc - msall.ic,
    }

    # Add the new columns
    msall = msall.assign(**new_columns2)

    # Write to csv
    msall.to_csv(output_dir + 'starall_HSCS21a_PI.csv', index=False)

    data_sg = apply_async_pool(8, grid,yedges, msall, xedges, yedges)
    Zsg_cl = Zs(data_sg, config_dict["data"]["pmracol"], config_dict["data"]["pmdeccol"])
    Zsg = Zs(data_sg, "pmra", "pmdec")
    Zra_g = pd.DataFrame(np.nanmean([Zra_g1,Zra_g2],axis=0))
    Zdec_g = pd.DataFrame(np.nanmean([Zdec_g1,Zdec_g2],axis=0))
    plot_corrmap(Zsg, Zsg_cl, Zra_g, Zdec_g, img_path, xedges, yedges)

    return 0

if __name__ == '__main__':
    base_dir = "/Users/akiratokiwa/workspace/Sextans_final/"
    parser = argparse.ArgumentParser(description='Galaxy correction')
    parser.add_argument('--star_path', type=str, default=f'{base_dir}catalog/product/HSCS21a_PI_star_cl.csv', help='star data directory')
    parser.add_argument('--gal_path', type=str, default=f'{base_dir}catalog/product/HSCS21a_PI_galaxy_cl.csv', help='galaxy data directory')
    parser.add_argument('--attenuation_path', type=str, default=f'{base_dir}catalog/HSC-SSP/HSCS21A_ext.csv', help='attenuation data directory')
    parser.add_argument('--output_dir', type=str, default=f'{base_dir}catalog/product/', help='output data directory')
    parser.add_argument('--img_path', type=str, default=f'{base_dir}img/plots/corrmap.png', help='output image directory')
    parser.add_argument('--config_path', type=str, default='/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/config.yaml', help='config directory')
    print("galaxy correction")
    main(**vars(parser.parse_args()))
