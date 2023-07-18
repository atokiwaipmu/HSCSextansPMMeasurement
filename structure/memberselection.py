"""
This script performs member selection operations on astronomical data related to stars and galaxies.

Author: Akira Tokiwa
"""

import argparse
import json
import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import interpolate
from scipy.optimize import curve_fit

# Append the path to the utils module
sys.path.append("/Users/akiratokiwa/Git/HSCSextansPMMeasurement")

from utils.utils import calR, expo, medians  # Import functions from the utils module



def plot_CMD(ms2: pd.DataFrame, mg_cl: pd.DataFrame, ms_core: pd.DataFrame, CMD_func: callable, giwidth: callable, xL: np.ndarray, img_path: str, config_dict: dict) -> None:
    """
    Plot the color-magnitude diagram (CMD).

    Args:
        ms2: DataFrame containing the ms2 data.
        mg_cl: DataFrame containing the mg_cl data.
        ms_core: DataFrame containing the ms_core data.
        CMD_func: Function for CMD.
        giwidth: Function for gi width.
        xL: Array for x values.
        img_path: Path where the image will be saved.
        config_dict: Configuration dictionary.
    """
    fig = plt.figure(figsize=(7, 5))
    ax0 = fig.add_axes([0, 0, 0.49, 1])
    ax0.scatter(ms2.gi, ms2.ic,color="black",alpha=0.1,s=2)

    ax0.set_xlabel(r'$\rm{(g - i)_{0}[mag]}$')
    ax0.set_ylabel(r'$\rm{i_{0, PSF}[mag]}$')
    ax0.text(-0.65,
            18,
            "All stars",
            horizontalalignment="left",
            verticalalignment="top", fontsize=16)
    ax0.axis([-0.75, 1.55, config_dict["data"]["i_max"], config_dict["data"]["i_min"]])

    ax1 = fig.add_axes([0.49, 0, 0.49, 1])
    tmp = mg_cl[calR(mg_cl.i_sdsscentroid_ra, mg_cl.i_sdsscentroid_dec, center=config_dict["prefix"]["center"]) < config_dict["process"]["r_core"]]
    ax1.scatter(tmp.gc - tmp.ic, tmp.ic, s=2, alpha=0.1, color="tab:red", label="Galaxy")
    ax1.scatter(ms_core.gi, ms_core.ic,color="black",alpha=0.1,s=2, label="Star")
    ax1.set_xlabel(r'$\rm{(g - i)_{0}[mag]}$')
    ax1.set_yticklabels([])
    ax1.text(-0.7,
            18,
            r"Core (R < $0.5^\circ$)",
            horizontalalignment="left",
            verticalalignment="top", fontsize=16)
    ax1.axis([-0.75, 1.55, config_dict["data"]["i_max"], config_dict["data"]["i_min"]])

    ax1.plot(CMD_func(xL)+giwidth(xL), xL,c="black",linestyle="-",lw=3)
    ax1.plot(CMD_func(xL)-giwidth(xL), xL,c="black",linestyle="-",lw=3)

    pL = np.linspace(config_dict["data"]["i_min"]+0.5, config_dict["data"]["i_max"]-0.5, 10)
    ax1.errorbar([-0.3]*10,pL,
                xerr = giwidth(pL),
                capsize=3,
                fmt='o',
                markersize=2,
                ecolor='black',
                markeredgecolor="black",
                color='cyan')
    
    fig.savefig(img_path, bbox_inches="tight")

def plot_giwidth(md_gierr: pd.DataFrame, errs_gi: pd.DataFrame, popt1: np.ndarray, popt2: np.ndarray, xL: np.ndarray, img_path: str) -> None:
    """
    Plot the gi width.

    Args:
        md_gierr: DataFrame containing the md_gierr data.
        errs_gi: DataFrame containing the errs_gi data.
        popt1: Array containing the optimal values for the parameters of popt1.
        popt2: Array containing the optimal values for the parameters of popt2.
        xL: Array for x values.
        img_path: Path where the image will be saved.
    """
    fig = plt.figure(figsize=(7, 4))
    ax1 = fig.add_axes([0, 0, 1, 1])
    ax1.scatter(xL,md_gierr, marker='^',s=20,color='tab:red')
    ax1.scatter(xL,errs_gi,c='tab:blue', marker="o", s=20)
    ax1.plot(xL, expo(xL,popt2[0], popt2[1]), color="tab:green",label = r"$1\sigma$ width of core CMD: $\sigma_\mathrm{CMD}$ ")
    ax1.plot(xL, expo(xL,popt1[0], popt1[1]), color="tab:orange", label = r"catalogue photometric error: $\sigma_{\left(g-i\right)}$")

    intri =  np.sqrt(expo(xL,popt2[0], popt2[1])**2-expo(xL,popt1[0], popt1[1])**2)
    ax1.plot(xL, intri + 3*expo(xL,popt1[0], popt1[1]), color="black", label =r"selection width: $\sqrt{\sigma_\mathrm{CMD}^2 - \sigma_{\left(g-i\right)}^2} + 3\sigma_{\left(g-i\right)}$")

    ax1.legend(loc="upper left", fontsize=14)
    ax1.set_xlabel(r'$\rm{i_{PSF}[mag]}$', fontsize=16)
    ax1.set_ylabel(r"color error $\sigma_{\left(g - i\right)}$ [mag]", fontsize=16)

    ax1.set_xlim(19.5, 24.05)
    fig.savefig(img_path, bbox_inches="tight")

def main(star_path: str, gal_path: str, attenuation_path: str, output_path: str, CMD_path: str, CMD_width_path: str, config_path: str, params_path: str) -> int:
    """
    Main function to perform member selection operations on the data.

    Args:
        star_path: Path to the input CSV file for stars.
        gal_path: Path to the input CSV file for galaxies.
        attenuation_path: Path to the input CSV file for attenuation.
        output_path: Path to the output CSV file.
        CMD_path: Path to the color-magnitude diagram (CMD) file.
        CMD_width_path: Path to the CMD width file.
        config_path: Path to the config file.
        params_path: Path to the params file.

    Returns:
        0 if the function runs successfully.
    """
    ms2=pd.read_csv(star_path)
    with open(config_path, 'r') as stream:
        config_dict = yaml.safe_load(stream)
    center, r_core  = config_dict["prefix"]["center"], config_dict["process"]["r_core"]
    racol, deccol = config_dict["data"]["racol"], config_dict["data"]["deccol"]
    magcol = config_dict["data"]["magcol"]
    
    # black list
    ms2 = ms2[ms2["# object_id"] != 42098376082216514]

    ms2.reset_index(drop=True, inplace=True)
    ms2["gierr"] = np.sqrt(ms2.g_psfflux_magerr**2 + ms2.i_psfflux_magerr**2)
    ms_core = ms2[((ms2[racol]- center[0])**2 + (ms2[deccol] - center[1])**2) < r_core**2]

    # pre-selection
    SGBRGB = ms_core[(ms_core.gi < 1) & (ms_core.gi > 0.1) & ((ms_core.gi > 0.7) | (ms_core.ic > 22) | (-3/30*(ms_core.ic-19)+0.7 < ms_core.gi))]
    
    # CMD width
    i_min, i_max, n_tmp = config_dict["data"]["i_min"], config_dict["data"]["i_max"], 30
    xL = np.linspace(i_min, i_max, n_tmp)
    md_gi, errs_gi = medians(SGBRGB, SGBRGB.ic, SGBRGB.gi, i_min, i_max, n_tmp, 0)
    md_gierr, errs_gierr = medians(SGBRGB, SGBRGB.ic, SGBRGB.gierr, i_min, i_max, n_tmp, 0)

    # fitting CMD width
    discard = 10 # discard 18 < i < 20
    index = (~np.isnan(md_gierr)) & (~np.isnan(errs_gi))
    popt1, pcov1 = curve_fit(expo, xL[index][discard:],md_gierr[index][discard:], sigma=errs_gierr[index][discard:])
    popt2, pcov2 = curve_fit(expo, xL[index][discard:],errs_gi[index][discard:])

    plot_giwidth(md_gierr, errs_gi, popt1, popt2, xL, CMD_width_path)

    CMD_func = interpolate.CubicSpline(xL,md_gi)

    def giwidth(ipsf, popt1=popt1, popt2=popt2):
        return np.sqrt(expo(ipsf,popt2[0], popt2[1])**2-expo(ipsf,popt1[0], popt1[1])**2)+3*expo(ipsf,popt1[0], popt1[1])
    
    CMD_func_ms2 = CMD_func(ms2[magcol])
    giwidth_ms2 = giwidth(ms2[magcol])

    ms_ccut = ms2[(ms2.gi < CMD_func_ms2 + giwidth_ms2) & (ms2.gi > CMD_func_ms2 - giwidth_ms2)]
    ms_ccut.to_csv(output_path, index=False)

    mg_cl = pd.read_csv(gal_path)
    m_ext=pd.read_csv(attenuation_path)
    mg_cl = mg_cl.merge(m_ext, on="# object_id")
    new_columns = {
        "gc": mg_cl.g_psfflux_mag - mg_cl.a_g,
        "rc": mg_cl.r_psfflux_mag - mg_cl.a_r,
        "ic": mg_cl.i_psfflux_mag - mg_cl.a_i,
    }
    mg_cl = mg_cl.assign(**new_columns)
    plot_CMD(ms2, mg_cl, ms_core, CMD_func, giwidth, xL, CMD_path, config_dict)

    if os.path.exists(params_path):
        with open(params_path) as f:
            params = json.load(f)
    else:
        params = {}

    params["gierr_func"] = {"core": popt1.tolist(), "catalogue": popt2.tolist()}

    with open(params_path, 'wt') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)    
    return 0

if __name__ == '__main__':
    base_dir = "/Users/akiratokiwa/workspace/Sextans_final/"
    parser = argparse.ArgumentParser(description='Member selection')
    parser.add_argument('--star_path', type=str, default=f'{base_dir}catalog/product/starall_HSCS21a_PI.csv', help='data directory')
    parser.add_argument('--gal_path', type=str, default=f'{base_dir}catalog/product/HSCS21a_PI_galaxy_cl.csv', help='data directory')
    parser.add_argument('--attenuation_path', type=str, default=f'{base_dir}catalog/HSC-SSP/HSCS21A_ext.csv', help='data directory')
    parser.add_argument('--output_path', type=str, default=f'{base_dir}catalog/product/starall_HSCS21a_PI_ccut.csv', help='data directory')
    parser.add_argument('--CMD_path', type=str, default=f'{base_dir}img/plots/CMD_membercut.png', help='data directory')
    parser.add_argument('--CMD_width_path', type=str, default=f'{base_dir}img/plots/CMD_width.pdf', help='data directory')
    parser.add_argument('--config_path', type=str, default='/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/config.yaml', help='config directory')
    parser.add_argument('--params_path', type=str, default='/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/params.json', help='config directory')
    print("memberselection.py")
    main(**vars(parser.parse_args()))
