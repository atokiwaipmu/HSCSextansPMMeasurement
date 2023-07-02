import pandas as pd
import numpy as np
import emcee
import os
import yaml
import argparse
import json

import sys
sys.path.append("/Users/akiratokiwa/Git/HSCSextansPMMeasurement")
from utils.utils import calarea, calR, proexpo, proplummer, proking, logprior, mcmc

def loglikelihood(x, inputs, prefix, priorset,model="plum", verbose=False):
    if model == "king":
        x0, y0, tmp_theta, eps, tmp_rh, tmp_rt= x
    else:
        x0, y0, tmp_theta, eps, tmp_rh= x
    theta = tmp_theta /180 * np.pi
    rh = tmp_rh /60 
    if model == "king":
        rt = tmp_rt /60
    
    ra, dec, Ntot = inputs
    area, sbg = prefix
    priormin, priormax = priorset
    if logprior(x, priormin, priormax, verbose=verbose) == -np.inf:
        if verbose:
            print("prior out of range")
        return -np.inf

    xi = (ra-x0)*np.cos(y0/180*np.pi)
    yi = (dec-y0)
    Ri = np.sqrt(((xi * np.cos(theta) - yi * np.sin(theta))/ (1 - eps))**2 +
                 (xi * np.sin(theta) + yi * np.cos(theta))**2)
    
    frac = (Ntot - area*sbg)
    if frac < 0:
        if verbose:
            print("frac < 0")
        return -np.inf
    if model == "plum":
        pm = proplummer(Ri, rh)/((1-eps)*np.pi*rh**2)
    elif model == "expo":
        pm = proexpo(Ri, rh)/((1-eps)*2*np.pi*rh**2)
    elif model == "king":
        a = rt/rh
        pm = proking(Ri[Ri < rt], rh, rt)/((1-eps)*np.pi*rh**2*(a**2/(1 + a**2) + 4/np.sqrt(1 + a**2) +np.log(1+a**2)-4))
        return np.sum(np.log(frac * pm + sbg)) + len(Ri[Ri > rt])* np.log(sbg)

    return np.sum(np.log(frac * pm + sbg))

def run_mcmc(priors, inputs, prefix, mcmc_type, max_n=10000, discard=2000, seed=1234, nwalkers=40):
    np.random.seed(seed)
    priorset = priors[0][mcmc_type], priors[1][mcmc_type]
    ndim = len(priorset[0])
    p0 = np.random.uniform(low=priorset[0], high=priorset[1], size=(nwalkers,ndim))
    sampler = mcmc(p0, ndim, nwalkers, loglikelihood, args=(inputs, prefix, priorset, mcmc_type), max_n=max_n)
    flat_samples = sampler.get_chain(discard=discard, flat=True)
    return flat_samples

def main(data_path, gal_path, distR_path, mid_dir, config_path, params_path, flag=False):
    ms_ccut = pd.read_csv(data_path)
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    racol, deccol = config_dict["data"]["racol"], config_dict["data"]["deccol"]
    center, r_tidal, areatick = config_dict["prefix"]["center"], config_dict["process"]["r_tidal"], config_dict["process"]["areatick"]
    data  = ms_ccut[calR(ms_ccut[racol], ms_ccut[deccol], center=center) < r_tidal].copy()
    ra, dec, Ntot = data[racol], data[deccol], len(data)

    if not os.path.exists(distR_path) or flag:
        xedges=np.arange(config_dict["data"]["xmin"], config_dict["data"]["xmax"], config_dict["process"]["areatick"])
        yedges=np.arange(config_dict["data"]["ymin"], config_dict["data"]["ymax"], config_dict["process"]["areatick"])
        ZdistR = calarea(gal_path, xedges, yedges, [center], True, False)
        np.savetxt(distR_path, ZdistR)
    else:
        ZdistR = np.loadtxt(distR_path)
    area = len(ZdistR[(ZdistR > 0)&(ZdistR < r_tidal)]) * (areatick**2)
    outer_area = len(ZdistR[(ZdistR > r_tidal)]) * (areatick**2)
    outer_nd = len(ms_ccut[calR(ms_ccut[racol], ms_ccut[deccol], center=center) > r_tidal]) / outer_area
    seed, nwalkers = config_dict["process"]["seed"], config_dict["process"]["nwalkers"]

    inputs = [ra,dec,Ntot]
    prefix = [area,outer_nd]

    initpmin_dict = {"plum": config_dict["structure"]["priormin"][:-1],
              "king": config_dict["structure"]["priormin"],
              "expo": config_dict["structure"]["priormin"][:-1]}
    initpmax_dict = {"plum": config_dict["structure"]["priormax"][:-1],
              "king": config_dict["structure"]["priormax"],
              "expo": config_dict["structure"]["priormax"][:-1]}

    samples = {}
    for mcmc_type in ["plum", "king", "expo"]:
        print(f"Start {mcmc_type} MCMC")
        samples[mcmc_type] = run_mcmc([initpmin_dict, initpmax_dict], inputs, prefix, mcmc_type, max_n=10000, discard=2000, seed=seed, nwalkers=nwalkers)

    # save sample series
    np.savetxt(mid_dir + "plum.txt", samples["plum"])
    np.savetxt(mid_dir + "king.txt", samples["king"])
    np.savetxt(mid_dir + "expo.txt", samples["expo"])

    # save bestfit
    if os.path.exists(params_path):
        with open(params_path) as f:
            params = json.load(f)
    else:
        params = {}
    for mcmc_type in ["plum", "king", "expo"]:
        bestfit = np.median(samples[mcmc_type], axis=0)
        params[mcmc_type]["bestfit"] = bestfit.tolist()
        params[mcmc_type]["bestfit_err"] = np.std(samples[mcmc_type], axis=0).tolist()
        params[mcmc_type]["bestfit_lll"] = loglikelihood(bestfit, inputs, prefix, [initpmin_dict[mcmc_type], initpmax_dict[mcmc_type]], model=mcmc_type, verbose=True)
    
    # choose the best model has the highest likelihood
    best_model = max(params, key=lambda x: params[x]["bestfit_lll"])
    params["chosen_model"] = best_model

    params["area"] = area
    params["outer_area"] = outer_area
    params["outer_nd"] = outer_nd

    with open(params_path, 'wt') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    base_dir = "/Users/akiratokiwa/workspace/Sextans_final/"
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=f'{base_dir}catalog/product/starall_HSCS21a_PI_ccut.csv')
    parser.add_argument('--gal_path', type=str, default=f'{base_dir}catalog/product/HSCS21a_PI_galaxy_cl.csv')
    parser.add_argument('--distR_path', type=str, default=f'{base_dir}params_estimated/ZdistR.txt')
    parser.add_argument('--mid_dir', type=str, default=f'{base_dir}params_estimated/')
    parser.add_argument('--config_path', type=str, default='/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/config.yaml')
    parser.add_argument('--params_path', type=str, default='/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/params.json')
    parser.add_argument('--flag', type=bool, default=False)
    print("structure.py")
    main(**vars(parser.parse_args()))