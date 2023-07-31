"""
This script performs proper motion measurements for celestial objects.

Author: Akira Tokiwa
Refined by: OpenAI's ChatGPT
"""

import numpy as np
import pandas as pd
import bisect
import scipy.interpolate as interpolate
import yaml
import argparse
import os
import json
import sys

# Assuming the utils module is in the same directory as this script
sys.path.append("/Users/akiratokiwa/Git/HSCSextansPMMeasurement")
from utils.utils import calRe, gaussian, logprior, mcmc

def chi2_1d(x, obs, bg, priorset):
    """
    Calculate the chi-square of the observed and modelled 1D data.

    Args:
        x: The model parameters.
        obs: The observed data.
        bg: The background data.
        priorset: The set of prior parameters.

    Returns:
        The chi-square of the observed and modelled data.
    """

    if len(x) == 2:
        pmra_sex, pmdec_sex = x
        pmra_bg, pmdec_bg = bg
    else:
        pmra_sex, pmdec_sex, pmra_bg, pmdec_bg = x

    # If the log prior is negative infinity, return it as the result
    if logprior(x, priorset[0], priorset[1]) == -np.inf:
        return -np.inf

    n_obs, pmra_obs, pmdec_obs, pmraerr, pmdecerr, n_MW = obs

    pmra_model = pmra_sex * (1 - n_MW) + pmra_bg * n_MW
    pmdec_model = pmdec_sex * (1 - n_MW) + pmdec_bg * n_MW

    pmraerr_n2 = pmraerr**2 + (pmra_sex - pmra_bg)**2 * (n_MW**2) / n_obs
    pmdecerr_n2 = pmdecerr**2 + (pmdec_sex - pmdec_bg)**2 * (n_MW**2) / n_obs

    result = np.sum((pmra_obs - pmra_model)**2 / pmraerr_n2 +
                    (pmdec_obs - pmdec_model)**2 / pmdecerr_n2)

    return -0.5 * result

def chi2_2d(x, inputs, prefix, numd_func, priorset):
    """
    Calculate the chi-square of the observed and modelled 2D data.

    Args:
        x: The model parameters.
        inputs: The observed data.
        prefix: The prefix parameters.
        numd_func: The number density function.
        priorset: The set of prior parameters.

    Returns:
        The chi-square of the observed and modelled data.
    """

    r, pmra, pmdec, pmraerr, pmdecerr = inputs

    if len(prefix) == 2:
        pmra_sex, pmdec_sex = x
        pmra_bg, pmdec_bg = prefix
    else:
        pmra_sex, pmdec_sex, pmra_bg, pmdec_bg = x

    # If the log prior is negative infinity, return it as the result
    if logprior(x, priorset[0], priorset[1]) == -np.inf:
        return -np.inf

    logli_ra = np.log(numd_func(r) * gaussian(pmra, pmra_sex, pmraerr) +
                      (1 - numd_func(r)) * gaussian(pmra, pmra_bg, pmraerr))

    logli_dec = np.log(numd_func(r) * gaussian(pmdec, pmdec_sex, pmdecerr) +
                       (1 - numd_func(r)) * gaussian(pmdec, pmdec_bg, pmdecerr))

    result = np.sum(logli_ra) + np.sum(logli_dec)

    return result

def run_mcmc(priors, loglikelihood, mcmc_type, *args, max_timestep=10000, discard=500, seed=1234, nwalkers=40):
    """
    Run the Markov Chain Monte Carlo (MCMC) process.

    Args:
        priors: The prior parameters.
        loglikelihood: The log likelihood function.
        mcmc_type: The type of MCMC to run.
        *args: Additional arguments for the log likelihood function.
        max_timestep: The maximum number of timesteps. Defaults to 10000.
        discard: The number of steps to discard. Defaults to 500.
        seed: The random seed. Defaults to 1234.
        nwalkers: The number of walkers. Defaults to 40.

    Returns:
        The flat samples from the MCMC process.
    """

    np.random.seed(seed)
    priorset = priors[0][mcmc_type], priors[1][mcmc_type]
    ndim = len(priorset[0])
    p0 = np.random.uniform(low=priorset[0], high=priorset[1], size=(nwalkers, ndim))

    sampler = mcmc(p0, ndim, nwalkers, loglikelihood, args=(*args, priorset), max_n=max_timestep)

    flat_samples = sampler.get_chain(discard=discard, flat=True)

    return flat_samples

def main(data_path,  rad_path, output1d_path, output2d_path, config_path, params_path):
    """
    Main function to perform the proper motion measurement.

    Args:
        data_path: Path to the input CSV file for data.
        rad_path: Path to the input numpy file for radius.
        output1d_path: Path to the output numpy file for 1D results.
        output2d_path: Path to the output numpy file for 2D results.
        config_path: Path to the configuration file.
        params_path: Path to the parameters file.
    """

    # Load the configuration and parameters files
    with open(params_path, "r") as f:
        params = json.load(f)

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Extract necessary parameters and column names
    model = params["chosen_model"]
    center_ra, center_dec, theta, eps = params[model]["bestfit"][:4]
    theta = theta / 180 * np.pi
    centers = np.array([center_ra, center_dec])
    rp = params["plum"]["bestfit"][4] / 60

    racol, deccol = config_dict["data"]["racol"], config_dict["data"]["deccol"]
    pmracol, pmdeccol = config_dict["data"]["pmracol"], config_dict["data"]["pmdeccol"]
    pmraerrcol, pmdecerrcol = config_dict["data"]["pmraerrcol"], config_dict["data"]["pmdecerrcol"]

    # Load the data and calculate the Re values
    ms_err = pd.read_csv(data_path)
    ms_err["Re"] = calRe(ms_err[racol], ms_err[deccol], centers, theta, eps)

    # Load the radial data
    radials = np.load(rad_path, allow_pickle=True)
    rs, r_all, r_pmra, r_pmraerr, r_pmdec, r_pmdecerr, r_numd, r_numderr, n_MW, r_area = radials

    r_num = r_numd * r_area
    numd_func = interpolate.CubicSpline(r_all, 1 - n_MW)

    n_2core = bisect.bisect(r_all, 2 * rp)
    n_end = len(r_num)

    n_end_dict = {
        "sxt_MW_free": len(r_num),
        "sxt_free": n_2core
    }

    initpmin_dict = {
        "sxt_MW_free": config_dict["PM"]["priormin"],
        "sxt_free": config_dict["PM"]["priormin"][:2]
    }

    initpmax_dict = {
        "sxt_MW_free": config_dict["PM"]["priormax"],
        "sxt_free": config_dict["PM"]["priormax"][:2]
    }

    seed, nwalkers, max_timestep = config_dict["process"]["seed"], config_dict["process"]["nwalkers"], config_dict["process"]["max_timestep"]

    samples_1d = {}
    bestfit_sxtMW = []

    # Run the MCMC process for each type
    for mcmc_type in ["sxt_MW_free", "sxt_free"]:
        n_end = n_end_dict[mcmc_type]
        obs = [r_num[1:n_end], r_pmra[1:n_end], r_pmdec[1:n_end], r_pmraerr[1:n_end],
               r_pmdecerr[1:n_end], np.array(n_MW)[1:n_end]]

        if mcmc_type == "sxt_free":
            bgr = [bestfit_sxtMW[0], bestfit_sxtMW[1]]
        else:
            bgr = []

        samples_1d[mcmc_type] = run_mcmc([initpmin_dict, initpmax_dict], chi2_1d, mcmc_type, obs, bgr, max_timestep=max_timestep, seed=seed, nwalkers=nwalkers)

        if mcmc_type == "sxt_MW_free":
            bestfit_sxtMW = np.median(samples_1d[mcmc_type], axis=0)[2:4]

    # Save the best fit parameters for each MCMC type
    for mcmc_type in ["sxt_MW_free", "sxt_free"]:
        params[mcmc_type] = {}
        params[mcmc_type]["bestfit_1d"] = np.median(samples_1d[mcmc_type], axis=0).tolist()
        params[mcmc_type]["bestfit_err_1d"] = np.std(samples_1d[mcmc_type], axis=0).tolist()

    # Save the 1D samples
    np.save(output1d_path, samples_1d)

    # Prepare the data for the 2D MCMC process
    data = ms_err[(ms_err.Re > rs[1]) & (ms_err.Re < 2 * rp)]
    inputs = [data.Re, data[pmracol], data[pmdeccol], data[pmraerrcol], data[pmdecerrcol]]

    samples_2d = {}

    # Run the MCMC process for each type
    for mcmc_type in ["sxt_MW_free", "sxt_free"]:
        if mcmc_type == "sxt_free":
            args = [inputs, bestfit_sxtMW, numd_func]
            samples_2d[mcmc_type] = run_mcmc([initpmin_dict, initpmax_dict], chi2_2d, mcmc_type, *args, max_timestep=max_timestep, seed=seed, nwalkers=nwalkers)
        else:
            args = [inputs, [], numd_func]
            samples_2d[mcmc_type] = run_mcmc([initpmin_dict, initpmax_dict], chi2_2d, mcmc_type, *args, max_timestep=max_timestep, seed=seed, nwalkers=nwalkers)

    # Save the 2D samples
    np.save(output2d_path, samples_2d)

    # Save the best fit parameters for each MCMC type
    for mcmc_type in ["sxt_MW_free", "sxt_free"]:
        params[mcmc_type]["bestfit_2d"] = np.median(samples_2d[mcmc_type], axis=0).tolist()
        params[mcmc_type]["bestfit_err_2d"] = np.std(samples_2d[mcmc_type], axis=0).tolist()

    # Save the parameters
    with open(params_path, 'wt') as f:
        json.dump(params, f, indent=2, ensure_ascii=False) 

if __name__ == "__main__":
    base_dir = "/Users/akiratokiwa/workspace/Sextans_final/"
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to the input CSV file for data.', default=f'{base_dir}catalog/product/starall_HSCS21a_PI_errorest.csv')
    parser.add_argument('--rad_path', required=True, help='Path to the input numpy file for radius.', default=f'{base_dir}params_estimated/numberdensity.npy')
    parser.add_argument('--output1d_path', required=True, help='Path to the output numpy file for 1D results.', default=f'{base_dir}params_estimated/samples_1d.npy')
    parser.add_argument('--output2d_path', required=True, help='Path to the output numpy file for 2D results.', default=f'{base_dir}params_estimated/samples_2d.npy')
    parser.add_argument('--config_path', required=True, help='Path to the configuration file.', default='/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/config.yaml')
    parser.add_argument('--params_path', required=True, help='Path to the parameters file.', default='/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/params.json')
    args = parser.parse_args()
    print("pmmeasurement.py")
    main(**vars(args))
