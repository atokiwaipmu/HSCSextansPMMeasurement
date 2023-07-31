"""
This script performs error estimation for proper motion measurements.

Author: Akira Tokiwa
Refined by: OpenAI's ChatGPT
"""

import argparse
import json
import numpy as np
import os
import pandas as pd
import yaml
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Assuming the utils module is in the same directory as this script
from utils.utils import calR, per1_sigma, bts, err_all, errmag, intri_fix

def calculate_errors(
    data: pd.DataFrame, 
    config_dict: dict, 
    min_count: int = 3
) -> pd.DataFrame:
    """
    Calculate errors based on the input data and configuration.

    Args:
        data: DataFrame containing the input data.
        config_dict: Dictionary containing the configuration parameters.
        min_count: Minimum count for error calculation. Defaults to 3.

    Returns:
        DataFrame with calculated errors.
    """
    # Extracting the necessary values from the configuration dictionary
    tick, mmin, mmax = config_dict["process"]["magtick"], config_dict["data"]["i_min"], config_dict["data"]["i_max"]
    bootstraps = config_dict["process"]["bootstraps"]
    pmracol, pmdeccol, magcol = config_dict["data"]["pmracol"], config_dict["data"]["pmdeccol"], config_dict["data"]["magcol"]

    # Initialize a list to store the data frames for different magnitude ranges
    imags = []

    # Generate the range of magnitudes to be considered
    im = np.arange(mmin, mmax, tick)

    # Count the number of data points in each magnitude bin
    counts = np.histogram(data[magcol], bins=im)[0]

    # Divide the data into different magnitude ranges and store them in the list
    for imag_low, imag_high in zip(im[:-1], im[1:]):
        imags.append(data[(data[magcol] > imag_low) & (data[magcol] < imag_high)])

    # Filter out the data frames with too few data points
    imags = [tmp for tmp in imags if len(tmp) > min_count]

    # Calculate the errors for each magnitude range
    pmraerr = [per1_sigma(tmp[pmracol]) for tmp in imags]
    pmdecerr = [per1_sigma(tmp[pmdeccol]) for tmp in imags]

    # Calculate the bootstrap standard errors for each magnitude range
    pmrabts = [np.std(np.apply_along_axis(per1_sigma, 1, bts(tmp[pmracol], bootstraps))) for tmp in imags]
    pmdecbts = [np.std(np.apply_along_axis(per1_sigma, 1, bts(tmp[pmdeccol], bootstraps))) for tmp in imags]

    # Compile the calculated errors into a data frame
    errs = pd.DataFrame(list(zip(im[:-1][counts > min_count], counts[counts > min_count], pmraerr, pmdecerr, pmrabts, pmdecbts)), 
                    columns=["imag", "num", "pmraerr", "pmdecerr", "pmrabts", "pmdecbts"])

    return errs


def fit_curve_and_apply(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    sigma_col: str, 
    p0: list = [20, 0.3, 20], 
    bounds: tuple = ((0, 0, 0), (np.inf, np.inf, np.inf))
) -> np.ndarray:
    """
    Fit a curve to the data and apply the curve fit.

    Args:
        df: DataFrame containing the data.
        x_col: Name of the x column.
        y_col: Name of the y column.
        sigma_col: Name of the sigma column.
        p0: Initial guess for the parameters. Defaults to [20, 0.3, 20].
        bounds: Bounds for the parameters. Defaults to ((0, 0, 0), (np.inf, np.inf, np.inf)).

    Returns:
        Optimal values for the parameters.
    """
    popt, pcov = curve_fit(err_all, df[x_col], df[y_col], p0=p0, bounds=bounds, sigma=df[sigma_col])
    return popt


def plot_error(
    errs: pd.DataFrame, 
    errsq: pd.DataFrame, 
    popt_ra: np.ndarray, 
    popt_dec: np.ndarray, 
    img_path: str
) -> None:
    """
    Plot the error.

    Args:
        errs: DataFrame containing the errs data.
        errsq: DataFrame containing the errsq data.
        popt_ra: Array containing the optimal values for the parameters of popt_ra.
        popt_dec: Array containing the optimal values for the parameters of popt_dec.
        img_path: Path where the image will be saved.
    """
    fig = plt.figure(figsize=(7, 7))
    ax0 = fig.add_axes([0, 0.5, 1, 0.5])

    # Plot the pmra errors with their bootstrap standard errors
    ax0.errorbar(
        errs.imag,
        errs.pmraerr,
        yerr=errs.pmrabts,
        capsize=6,
        fmt='o',
        markersize=8,
        color='tab:blue',
        alpha=0.6
    )

    # Plot the fitted curve for the pmra errors
    ax0.plot(
        errs.imag,
        err_all(errs.imag, popt_ra[0], popt_ra[1], popt_ra[2]),
        c="tab:orange",
        label="PM dispersion",
        alpha=0.8,
        lw=3
    )

    # Plot the intrinsic dispersion for the pmra errors
    ax0.plot(
        errs.imag,
        intri_fix(errs.imag, popt_ra[2]),
        linestyle="-.",
        c="tab:green",
        label="intrinsic dispersion",
        alpha=0.8,
        lw=3
    )

    # Plot the photometric error for the pmra errors
    ax0.plot(
        errs.imag,
        errmag(errs.imag, popt_ra[0], popt_ra[1]),
        linestyle="--",
        c="tab:red",
        label="photometric error",
        alpha=0.8,
        lw=3
    )

    # Set the y-axis label
    ax0.set_ylabel(r"$\\sigma_{\\mu_\\alpha}$" + ' [mas/yr]', fontsize=20)

    # Set the y-axis limits and hide the x-axis labels
    ax0.set_ylim(-0.50, 14)
    ax0.set_xticklabels([])

    # Plot the quasar pmra errors with their bootstrap standard errors
    ax0.errorbar(
        errsq.imag,
        errsq.pmraerr,
        yerr=errsq.pmrabts, 
        capsize=6,
        markersize=8,
        label="quasar",
        fmt='^',
        color="tab:pink",
        alpha=0.6
    )

    # Create a new axis for the pmdec errors
    ax1 = fig.add_axes([0, 0, 1, 0.5])

    # Plot the pmdec errors with their bootstrap standard errors
    ax1.errorbar(
        errs.imag,
        errs.pmdecerr,
        yerr=errs.pmdecbts,
        capsize=6,
        fmt='o',
        markersize=8,
        color='tab:blue',
        alpha=0.6
    )

    # Plot the fitted curve for the pmdec errors
    ax1.plot(
        errs.imag,
        err_all(errs.imag, popt_dec[0], popt_dec[1], popt_dec[2]),
        c="tab:orange",
        label="PM variance",
        alpha=0.8,
        lw=3
    )

    # Plot the intrinsic dispersion for the pmdec errors
    ax1.plot(
        errs.imag,
        intri_fix(errs.imag, popt_dec[2]),
        linestyle="-.",
        c="tab:green",
        label="intrinsic dispersion",
        alpha=0.8,
        lw=3
    )

    # Plot the photometric error for the pmdec errors
    ax1.plot(
        errs.imag,
        errmag(errs.imag, popt_dec[0], popt_dec[1]),
        linestyle="--",
        c="tab:red",
        label="photometric error",
        alpha=0.8,
        lw=3
    )

    # Plot the quasar pmdec errors with their bootstrap standard errors
    ax1.errorbar(
        errsq.imag,
        errsq.pmdecerr,
        yerr=errsq.pmdecbts, 
        capsize=6,
        markersize=8,
        label="quasar",
        fmt='^',
        color="tab:pink",
        alpha=0.6
    )

    # Set the x-axis and y-axis labels
    ax1.set_xlabel(r'$\\rm{i_{psf}}$' + ' [mag]', fontsize=20)
    ax1.set_ylabel(r"$\\sigma_{\\mu_\\delta}$" + ' [mas/yr]', fontsize=20)

    # Set the y-axis limits
    ax1.set_ylim(-0.50, 14)

    # Create a legend
    ax1.legend(fontsize=15, loc="upper left")

    # Save the figure
    plt.savefig(img_path, dpi=300, bbox_inches='tight')

def main(
    data_path: str, 
    quasar_path: str, 
    output_path: str, 
    img_path: str, 
    config_path: str, 
    params_path: str
) -> None:
    """
    Main function to run the error estimation process.

    Args:
        data_path: Path to the input file.
        quasar_path: Path to the quasar file.
        output_path: Path to the output file.
        img_path: Path to save the output image.
        config_path: Path to the configuration file.
        params_path: Path to the parameters file.
    """
    # Load the configuration file
    with open(config_path, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    # Load the data
    data = pd.read_csv(data_path)

    # Calculate the errors
    errs = calculate_errors(data, config_dict)

    # Fit the curves and apply
    popt_ra = fit_curve_and_apply(errs, "imag", "pmraerr", "pmrabts")
    popt_dec = fit_curve_and_apply(errs, "imag", "pmdecerr", "pmdecbts")

    # Load the quasar data
    quasar_data = pd.read_csv(quasar_path)

    # Calculate the quasar errors
    errsq = calculate_errors(quasar_data, config_dict)

    # Fit the quasar curves and apply
    popt_qra = fit_curve_and_apply(errsq, "imag", "pmraerr", "pmrabts")
    popt_qdec = fit_curve_and_apply(errsq, "imag", "pmdecerr", "pmdecbts")

    # Plot the error
    plot_error(errs, errsq, popt_ra, popt_dec, img_path)

    # Save the results
    data.to_csv(output_path, index=False)

    # Load or initialize the parameters
    if os.path.exists(params_path):
        with open(params_path) as f:
            params = json.load(f)
    else:
        params = {}

    # Update and save the parameters
    params["errorest"] = {"pmra": popt_ra.tolist(), "pmdec": popt_dec.tolist()}
    with open(params_path, 'wt') as f:
        json.dump(params, f, indent=2, ensure_ascii=False) 

if __name__ == '__main__':
    base_dir = "/Users/akiratokiwa/workspace/Sextans_final/"
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help="Path to the input file.", default=f'{base_dir}catalog/product/starall_HSCS21a_PI_ccut.csv')
    parser.add_argument('--quasar_path', required=True, help="Path to the quasar file.", default=f'{base_dir}catalog/product/HSCS21a_SDSSQSO_pm.csv')
    parser.add_argument('--output_path', required=True, help="Path to the output file.", default=f'{base_dir}catalog/product/starall_HSCS21a_PI_errorest.csv')
    parser.add_argument('--img_path', required=True, help="Path to save the output image.", default=f'{base_dir}img/plots/errorest.pdf')
    parser.add_argument('--config_path', required=True, help="Path to the configuration file.", default="/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/config.yaml")
    parser.add_argument('--params_path', required=True, help="Path to the parameters file.", default="/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/params.json")
    args = parser.parse_args()
    
    main(args.data_path, args.quasar_path, args.output_path, args.img_path, args.config_path, args.params_path)