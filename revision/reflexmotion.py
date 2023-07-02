
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
import astropy.coordinates as coord
import matplotlib.pyplot as plt
import yaml
import argparse
import os
import json

import sys
sys.path.append("/Users/akiratokiwa/Git/HSCSextansPMMeasurement/")
from utils.utils import wmeans

def solarmotion(x, y, dist=10):
    pos = x, y
    pm = [0, 0]
    vr = 0
    state = coord.galactocentric_frame_defaults.get_from_registry("v4.0")

    c1 = SkyCoord(ra=pos[0]*u.degree, dec=pos[1]*u.degree, distance=dist*u.kpc, pm_ra_cosdec=pm[0]*u.mas/u.year, pm_dec=pm[1]*u.mas/u.year, radial_velocity=vr*u.km/u.s, frame='icrs')
    c2 = c1.transform_to("galactocentric")
    rep = c2.cartesian.without_differentials()
    c3 = rep.with_differentials(c2.cartesian.differentials['s'] - state["parameters"][ 'galcen_v_sun'])
    c4 = SkyCoord(c3, frame="galactocentric").transform_to("icrs")
    
    return [c4.pm_ra_cosdec.value, c4.pm_dec.value]

def create_axis(ax, t_pro, edges, magmaxs, xlabel, ylabel, label_prefix, mean_values, bg, axis, dim):
    for i, t in enumerate(t_pro):
        ax.errorbar(edges, t[0], yerr=t[1], fmt="o", label="{} {}".format(label_prefix, magmaxs[i]), color="tab:blue")
    if axis == "ra":
        ax.plot(edges, np.median(mean_values, axis=0), label="8 kpc reflex motion gradient", color="tab:green")
    else:
        ax.plot(edges, np.median(mean_values, axis=1), label="8 kpc reflex motion gradient", color="tab:green")
    if dim == "ra":
        ax.set_ylim(-6, 1)
    else:
        ax.set_ylim(-8, 0)
    ax.hlines(bg, edges[0], edges[-1], color="tab:orange", label="fixed background: {:.3f}".format(bg))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def main(data_path, img_path, config_path, params_path):
    ms_err = pd.read_csv(data_path)

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    racol, deccol = config_dict["data"]["racol"], config_dict["data"]["deccol"]
    pmracol, pmdeccol = config_dict["data"]["pmracol"], config_dict["data"]["pmdeccol"]
    pmraerrcol, pmdecerrcol = config_dict["data"]["pmraerrcol"], config_dict["data"]["pmdecerrcol"]
    magcol = config_dict["data"]["magcol"]
    r_tidal = config_dict["process"]["r_tidal"]

    with open(params_path) as f:
        params = json.load(f)
    bgs = params["sxt_MW_free"]["bestfit_1d"][2:4]

    xmin, xmax = config_dict["data"]["xmin"], config_dict["data"]["xmax"]
    ymin, ymax = config_dict["data"]["ymin"], config_dict["data"]["ymax"]
    tick = 0.25
    xedges = np.arange(xmin,xmax,tick)
    yedges = np.arange(ymin,ymax,tick)
    xx, yy = np.meshgrid(xedges, yedges)

    magmaxs = [20.5]
    ts = [ms_err[(ms_err.R > r_tidal) & (ms_err[magcol] < magmax)] for magmax in magmaxs]
    alls = ms_err[(ms_err.R > r_tidal)]
                  
    a10 = np.zeros(xx.shape)
    b10 = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            a10[i][j], b10[i][j] = solarmotion(xx[i][j], yy[i][j], 8)

    edges_map = {"ra": xedges, "dec": yedges}
    centroid_map = {"ra": racol, "dec": deccol}
    motion_map = {"ra": pmracol, "dec": pmdeccol}
    motionerr_map = {"ra": pmraerrcol, "dec": pmdecerrcol}
    limits_map = {"ra": (xmin, xmax), "dec": (ymin, ymax)}
    mean_values_map = {"ra": a10, "dec": b10}
    ylabels_map = {"ra": r"$\mu_{\alpha*}$ [mas/yr]", "dec": r"$\mu_{\delta}$ [mas/yr]"}
    xlabels_map = {"ra": r"RA [deg]", "dec": r"Dec. [deg]"}

    t_pros = {}
    for axis in ["ra", "dec"]:
        t_pros[axis] = {}
        for dim in ["ra", "dec"]:
            t_pros[axis][dim] = [wmeans(t, t[centroid_map[axis]], t[motion_map[dim]], t[motionerr_map[dim]], *limits_map[axis], len(edges_map[axis])) for t in ts]

    fig, ax = plt.subplots(2,2, figsize=(10,10))
    for i, axis in enumerate(["ra", "dec"]):
        for j, dim in enumerate(["ra", "dec"]):
            ax[i,j] = create_axis(ax[i,j], t_pros[axis][dim], edges_map[axis], magmaxs, xlabels_map[axis],  ylabels_map[dim], "i <", mean_values_map[dim], bgs[j], axis, dim)
            all_wm= wmeans(alls, alls[centroid_map[axis]], alls[motion_map[dim]], alls[motionerr_map[dim]], *limits_map[axis], len(edges_map[axis]))
            ax[i,j].errorbar(edges_map[axis], all_wm[0], yerr=all_wm[1], fmt="o", label="all magnitude", color="tab:red", alpha=0.4)
            ax[i,j].legend(loc="upper left")

    fig.savefig(img_path, dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    base_dir = "/Users/akiratokiwa/workspace/Sextans_final/"
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=f'{base_dir}catalog/product/starall_HSCS21a_PI_errorest.csv')
    parser.add_argument('--img_path', type=str, default=f'{base_dir}img/plots/reflexmotion.png')
    parser.add_argument('--config_path', type=str, default='/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/config.yaml')
    parser.add_argument('--params_path', type=str, default='/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/params.json')
    args = parser.parse_args()
    main(**vars(args))