import numpy as np
import pandas as pd
import argparse
import os
import yaml
import json

import sys
sys.path.append("/Users/akiratokiwa/Git/HSCSextansPMMeasurement")
from utils.utils import calRe, wmean, wmeanerr, calarea

def calculate_r(radius, rs, min_count=300):
    counts = np.histogram(radius, bins=rs)[0]
    idxs = []
    idx=0
    while np.sum(counts)>min_count:
        idx = np.where(np.cumsum(counts) // min_count > 0)[0][0]
        counts[:idx] = 0
        idxs.append(idx)
    return rs[idxs]

def calculate_radius(data, rs):
    return [data[(data.Re > r_low) & (data.Re <= r_high)] for r_low, r_high in zip(rs[:-1], rs[1:])]

def calculate_r_area(Zdist2d, rs, areatick=0.025):
    return [np.sum(Zdist2d[(Zdist2d <= r_high) & (Zdist2d > r_low)].count()) * (areatick**2) for r_low, r_high in zip(rs[:-1], rs[1:])]

def main(data_path, gal_path, distRe_path, output_path, config_path, params_path, flag=False):
    with open(params_path, "r") as f:
        params = json.load(f)
    outer_nd = params["outer_nd"]
    model = params["chosen_model"]
    center_ra, center_dec, theta, eps = params[model]["bestfit"][:4]
    theta = theta /180 * np.pi
    centers = np.array([center_ra, center_dec])

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    racol, deccol = config_dict["data"]["racol"], config_dict["data"]["deccol"]
    pmracol, pmdeccol = config_dict["data"]["pmracol"], config_dict["data"]["pmdeccol"]
    pmraerrcol, pmdecerrcol = config_dict["data"]["pmraerrcol"], config_dict["data"]["pmdecerrcol"]

    ms_err = pd.read_csv(data_path)
    ms_err["Re"] = calRe(ms_err[racol], ms_err[deccol], centers, theta, eps)

    rs = calculate_r(ms_err.Re, np.arange(0,4,0.001))
    if not os.path.exists(distRe_path) or flag:
        xedges=np.arange(config_dict["data"]["xmin"], config_dict["data"]["xmax"], config_dict["process"]["areatick"])
        yedges=np.arange(config_dict["data"]["ymin"], config_dict["data"]["ymax"], config_dict["process"]["areatick"])
        Zdist2d = calarea(gal_path, xedges, yedges, [centers, theta, eps], False, True)
        np.savetxt(distRe_path, Zdist2d)
        Zdist2d = pd.DataFrame(Zdist2d)
    else:
        Zdist2d = pd.DataFrame(np.loadtxt(distRe_path))
        
    r_area = np.array(calculate_r_area(Zdist2d, rs, areatick=config_dict["process"]["areatick"]))
    radius = calculate_radius(ms_err, rs)
    r_all = np.array([radius_chunk.Re.mean() for radius_chunk in radius])

    r_num = np.array([len(radius_chunk) for radius_chunk in radius])
    r_numd = r_num/r_area
    r_numderr =  np.sqrt(r_num)/r_area
    n_MW = outer_nd / r_numd

    n_MW = np.where(n_MW > 1, 1, n_MW)
    n_MW[np.argmax(n_MW):] = 1
    r_pmra = np.array([wmean(rad[pmracol], rad[pmraerrcol]) for rad in radius])
    r_pmraerr = np.array([wmeanerr(rad[pmraerrcol]) for rad in radius])
    r_pmdec = np.array([wmean(rad[pmdeccol], rad[pmdecerrcol]) for rad in radius])
    r_pmdecerr = np.array([wmeanerr(rad[pmdecerrcol]) for rad in radius])

    data = np.array([rs, r_all, r_pmra, r_pmraerr, r_pmdec, r_pmdecerr, r_numd, r_numderr, n_MW, r_area])
    np.save(output_path, data)

if __name__ == "__main__":
    base_dir = "/Users/akiratokiwa/workspace/Sextans_final/"
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=f'{base_dir}catalog/product/starall_HSCS21a_PI_errorest.csv')
    parser.add_argument('--gal_path', type=str, default=f'{base_dir}catalog/product/HSCS21a_PI_galaxy_cl.csv')
    parser.add_argument('--distRe_path', type=str, default=f'{base_dir}params_estimated/ZdistRe.txt')
    parser.add_argument('--output_path', type=str, default=f'{base_dir}params_estimated/numberdensity.npy')
    parser.add_argument('--config_path', type=str, default='/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/config.yaml')
    parser.add_argument('--params_path', type=str, default='/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/params.json')
    args = parser.parse_args()
    main(**vars(args))