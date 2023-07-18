"""
This script processes astronomical data by applying various transformations, 
computations, and removal of certain data points. It also plots the data and 
saves the processed data into a CSV file.

Author: Akira Tokiwa
"""

import argparse
import sys
from typing import Tuple

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Append the path to the utils module
sys.path.append("/Users/akiratokiwa/Git/HSCSextansPMMeasurement")

from utils.utils import clip  # Import the clip function from the utils module


def plot_giclip(df: pd.DataFrame, df_med: pd.DataFrame, df_sigG: pd.DataFrame, i_min: float, i_max: float, img_path: str) -> None:
    """
    Plot the g-i color Clip and save the figure.
    
    Args:
        df: DataFrame containing the data.
        df_med: DataFrame containing the median data.
        df_sigG: DataFrame containing the standard deviation data.
        i_min: The minimum value for i flux.
        i_max: The maximum value for i flux.
        img_path: Path where the image will be saved.
    """
    # Creating the figure and the axes
    fig = plt.figure(figsize=(7, 7))
    ax0 = fig.add_axes([0, 0, 1, 0.85])
    ax2 = fig.add_axes([0, 0.96, 1, 0.04])

    # Define the edges and the norm
    xedges = np.arange(i_min, i_max, 0.1)
    yedges = np.arange(-4., 4.1, 0.1)
    norm = colors.LogNorm(vmin=1, vmax=50000)

    SSP_gi = df.g_psfflux_mag - df.i_psfflux_mag
    PI_gi = df.g_psfflux - df.i_psfflux
    hist = np.histogram2d(df.i_psfflux_mag, PI_gi - SSP_gi, bins=(xedges, yedges))

    # Plotting the data
    ax0.plot(np.arange(17, 24, 0.1)[df_med != 0], df_med[df_med != 0] + 3 * df_sigG[0][df_med != 0], lw=2, c="cyan")
    ax0.plot(np.arange(17, 24, 0.1)[df_med != 0], df_med[df_med != 0] - 3 * df_sigG[1][df_med != 0], lw=2, c="cyan")

    z = hist[0].T
    mappable = ax0.pcolormesh(xedges, yedges, z, norm=norm, cmap="hot")

    # Setting labels
    ax0.set_xlabel('$i_\\mathrm{PSF}$', fontsize=16)
    ax0.set_ylabel('$(g-i)_\\mathrm{PI}-(g-i)_\\mathrm{SSP}$', fontsize=16)
    ax0.text(18, 3.5, "Star", size=16, horizontalalignment="left", verticalalignment="top")

    # Adding the colorbar
    fig.colorbar(mappable, aspect=40, shrink=0.6, orientation='horizontal', extend='both', cax=ax2)

    # Save the figure
    fig.savefig(img_path, bbox_inches="tight")


def replace_inf_nan_dropna(df: pd.DataFrame, col_list: list) -> pd.DataFrame:
    """
    Replace infinite and NaN values, and drop the rows with missing values.

    Args:
        df: DataFrame containing the data.
        col_list: List of column names to check for missing values.

    Returns:
        The processed DataFrame.
    """
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=col_list, inplace=True)
    return df


def compute_pm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the proper motion.

    Args:
        df: DataFrame containing the data.

    Returns:
        The DataFrame with the proper motion computed.
    """
    df['delta_yr'] = (df.HSC_mjd - df.mjd_x) / 365.24
    df['dra'] = (df.i_sdsscentroid_ra - df.i_ra) * 3600 * 1000  # mas
    df['ddec'] = (df.i_sdsscentroid_dec - df.i_dec) * 3600 * 1000  # mas
    df['pmra'] = df.dra / df.delta_yr  # mas/yr
    df['pmdec'] = df.ddec / df.delta_yr  # mas/yr
    return df


def compute_color(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the color.

    Args:
        df: DataFrame containing the data.

    Returns:
        The DataFrame with the color computed.
    """
    df["gr"] = df.g_psfflux_mag - df.r_psfflux_mag
    df["ri"] = df.r_psfflux_mag - df.i_psfflux_mag
    df["gi"] = df.g_psfflux_mag - df.i_psfflux_mag
    return df


def remove_sextans_C(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the sextans C from the DataFrame.

    Args:
        df: DataFrame containing the data.

    Returns:
        The DataFrame with sextans C removed.
    """
    df = df[(df.i_sdsscentroid_ra > 151.43) | (df.i_sdsscentroid_ra < 151.33) | (df.i_sdsscentroid_dec > 0.13) | (df.i_sdsscentroid_dec < 0.02)].copy()
    return df


def process_dataframe(data_path: str, img_path: str, output_path: str, starflag: bool = False) -> int:
    """
    Read and process the data from the CSV file, plot the data, and save the processed data into a CSV file.

    Args:
        data_path: Path to the input CSV file.
        img_path: Path where the image will be saved.
        output_path: Path to the output CSV file.
        starflag: Flag to indicate whether to compute color and remove sextans C. Defaults to False.

    Returns:
        0 if the function runs successfully.
    """
    # Read data
    df = pd.read_csv(data_path)
    col_list = ['i_sdsscentroid_ra', 'i_sdsscentroid_dec', "mjd_x", "HSC_mjd", "i_psfflux_mag", "g_psfflux_mag", "i_psfflux", "g_psfflux"]
    df = replace_inf_nan_dropna(df, col_list)

    # Clip
    i_min, i_max = [17, 24]
    SSP_gi = df.g_psfflux_mag - df.i_psfflux_mag
    PI_gi = df.g_psfflux - df.i_psfflux
    past_df = df.copy()
    df, df_med, df_sigG = clip(df, df.i_psfflux_mag, PI_gi - SSP_gi, i_min, i_max, (i_max - i_min) * 10, 0)
    plot_giclip(past_df, df_med, df_sigG, i_min, i_max, img_path)

    # Compute pm
    df = compute_pm(df)
    if starflag:
        df = compute_color(df)
        df = remove_sextans_C(df)
    df.to_csv(output_path, index=False)

    return 0


def main(data_dir: str, img_dir: str, output_dir: str) -> None:
    """
    Main function to process the data for both stars and galaxies.

    Args:
        data_dir: Directory path to the input CSV files.
        img_dir: Directory path to the output image files.
        output_dir: Directory path to the output CSV files.
    """
    star_dirs = {"data": f'{data_dir}matched_HSCS21a_PI_star.csv',
                 "img": f'{img_dir}/colorclip_star.png',
                 "output": f'{output_dir}HSCS21a_PI_star_cl.csv'}

    galaxy_dirs = {"data": f'{data_dir}matched_HSCS21a_PI_galaxy.csv',
                   "img": f'{img_dir}colorclip_galaxy.png',
                   "output": f'{output_dir}HSCS21a_PI_galaxy_cl.csv'}

    process_dataframe(star_dirs["data"], star_dirs["img"], star_dirs["output"], starflag=True)
    process_dataframe(galaxy_dirs["data"], galaxy_dirs["img"], galaxy_dirs["output"], starflag=False)


if __name__ == '__main__':
    base_dir = "/Users/akiratokiwa/workspace/Sextans_final/"
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=f'{base_dir}catalog/matched/')
    parser.add_argument('--img_dir', type=str, default=f'{base_dir}img/plots/')
    parser.add_argument('--output_dir', type=str, default=f'{base_dir}catalog/product/')
    args = parser.parse_args()
    print("dataprocess.py")
    main(args.data_dir, args.img_dir, args.output_dir)
