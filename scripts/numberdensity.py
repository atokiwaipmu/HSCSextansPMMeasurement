"""
This script calculates the number density of celestial objects.

Author: Akira Tokiwa
Refined by: OpenAI's ChatGPT
"""

import numpy as np
import pandas as pd
import argparse
import os
import yaml
import json
import sys

# Assuming the utils module is in the same directory as this script
sys.path.append("/Users/akiratokiwa/Git/HSCSextansPMMeasurement")
from utils.utils import calRe, wmean, wmeanerr, calarea

def calculate_radius_bins(radius: pd.Series, rs: np.ndarray, min_count: int = 300) -> np.ndarray:
    """
    Calculate the radius bins with a minimum count.

    Args:
        radius: Series containing the radius values.
        rs: Array containing the radius bin edges.
        min_count: Minimum count for a bin to be considered.

    Returns:
        Array containing the bin edges that meet the minimum count requirement.
    """
    counts = np.histogram(radius, bins=rs)[0]
    idxs = []
    idx = 0
    while np.sum(counts) > min_count:
        idx = np.where(np.cumsum(counts) // min_count > 0)[0][0]
        counts[:idx] = 0
        idxs.append(idx)
    return rs[idxs]

def split_data_into_radius_bins(data: pd.DataFrame, rs: np.ndarray) -> list:
    """
    Split the data into different radius bins.

    Args:
        data: DataFrame containing the data.
        rs: Array containing the radius bin edges.

    Returns:
        List of DataFrames, each containing the data for a specific radius bin.
    """
    return [data[(data.Re > r_low) & (data.Re <= r_high)] for r_low, r_high in zip(rs[:-1], rs[1:])]

def calculate_bin_areas(Zdist2d: pd.DataFrame, rs: np.ndarray, areatick: float = 0.025) -> list:
    """
    Calculate the area of each bin.

    Args:
        Zdist2d: DataFrame containing the Zdist2d values.
        rs: Array containing the radius bin edges.
        areatick: Area tick size. Defaults to 0.025.

    Returns:
        List containing the area of each bin.
    """
    return [np.sum(Zdist2d[(Zdist2d <= r_high) & (Zdist2d > r_low)].count()) * (areatick**2) for r_low, r_high in zip(rs[:-1], rs[1:])]

def main(
    data_path: str, 
    gal_path: str, 
    distRe_path: str, 
    output_path: str, 
    config_path: str, 
    params_path: str, 
    flag: bool = False
) -> None:
    """
    Main function to perform the number density calculation.

    Args:
        data_path: Path to the input CSV file for data.
        gal_path: Path to the input CSV file for galaxies.
        distRe_path: Path to the input text file for distRe.
        output_path: Path to the output numpy file.
        config_path: Path to the config file.
        params_path: Path to the params file.
        flag: A flag to indicate whether to recalculate distRe. Defaults to False.
    """
    # Load the configuration and parameters files
    with open(params_path, "r") as f:
        params = json.load(f)
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Extract necessary parameters and column names
    outer_nd = params["outer_nd"]
    model = params["chosen_model"]
    center_ra, center_dec, theta, eps = params[model]["bestfit"][:4]
    theta = theta /180 * np.pi
    centers = np.array([center_ra, center_dec])

    racol, deccol = config_dict["data"]["racol"], config_dict["data"]["deccol"]
    pmracol, pmdeccol = config_dict["data"]["pmracol"], config_dict["data"]["pmdeccol"]
    pmraerrcol, pmdecerrcol = config_dict["data"]["pmraerrcol"], config_dict["data"]["pmdecerrcol"]

    # Load the data and calculate the Re values
    data = pd.read_csv(data_path)
    data["Re"] = calRe(data[racol], data[deccol], centers, theta, eps)

    # Calculate the radius bins
    rs = calculate_radius_bins(data.Re, np.arange(0, 4, 0.001))

    # If flag is True or distRe file doesn't exist, calculate and save distRe
    if flag or not os.path.exists(distRe_path):
        xedges = np.arange(config_dict["data"]["xmin"], config_dict["data"]["xmax"], config_dict["process"]["areatick"])
        yedges = np.arange(config_dict["data"]["ymin"], config_dict["data"]["ymax"], config_dict["process"]["areatick"])
        Zdist2d = calarea(gal_path, xedges, yedges, [centers, theta, eps], False, True)
        np.savetxt(distRe_path, Zdist2d)
        Zdist2d = pd.DataFrame(Zdist2d)
    else:
        Zdist2d = pd.DataFrame(np.loadtxt(distRe_path))

    # Calculate the area of each bin
    r_area = np.array(calculate_bin_areas(Zdist2d, rs, areatick=config_dict["process"]["areatick"]))

    # Split the data into different radius bins
    radius_bins = split_data_into_radius_bins(data, rs)

    # Calculate the mean radius for each bin
    r_all = np.array([bin_data.Re.mean() for bin_data in radius_bins])

    # Calculate the number of data points in each bin
    r_num = np.array([len(bin_data) for bin_data in radius_bins])

    # Calculate the number density and its error for each bin
    r_numd = r_num / r_area
    r_numderr =  np.sqrt(r_num) / r_area

    # Calculate the Milky Way number density for each bin
    n_MW = outer_nd / r_numd
    n_MW = np.where(n_MW > 1, 1, n_MW)
    n_MW[np.argmax(n_MW):] = 1

    # Calculate the weighted mean and its error for pmra and pmdec for each bin
    r_pmra = np.array([wmean(bin_data[pmracol], bin_data[pmraerrcol]) for bin_data in radius_bins])
    r_pmraerr = np.array([wmeanerr(bin_data[pmraerrcol]) for bin_data in radius_bins])
    r_pmdec = np.array([wmean(bin_data[pmdeccol], bin_data[pmdecerrcol]) for bin_data in radius_bins])
    r_pmdecerr = np.array([wmeanerr(bin_data[pmdecerrcol]) for bin_data in radius_bins])

    # Create an array to save all the calculated data
    data = np.array([rs, r_all, r_pmra, r_pmraerr, r_pmdec, r_pmdecerr, r_numd, r_numderr, n_MW, r_area])

    # Save the data
    np.save(output_path, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to the input data file.', default='/Users/akiratokiwa/workspace/Sextans_final/catalog/product/starall_HSCS21a_PI_errorest.csv')
    parser.add_argument('--gal_path', required=True, help='Path to the input galaxy file.', default='/Users/akiratokiwa/workspace/Sextans_final/catalog/product/HSCS21a_PI_galaxy_cl.csv')
    parser.add_argument('--distRe_path', required=True, help='Path to the input distRe file.', default='/Users/akiratokiwa/workspace/Sextans_final/params_estimated/ZdistRe.txt')
    parser.add_argument('--output_path', required=True, help='Path to the output file.', default='/Users/akiratokiwa/workspace/Sextans_final/params_estimated/numberdensity.npy')
    parser.add_argument('--config_path', required=True, help='Path to the configuration file.', default='/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/config.yaml')
    parser.add_argument('--params_path', required=True, help='Path to the parameters file.', default='/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/params.json')
    args = parser.parse_args()
    
    main(
        data_path=args.data_path, 
        gal_path=args.gal_path, 
        distRe_path=args.distRe_path, 
        output_path=args.output_path, 
        config_path=args.config_path, 
        params_path=args.params_path
    )