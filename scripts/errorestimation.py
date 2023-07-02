import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import argparse
import yaml
import json
import os

import sys
sys.path.append("/Users/akiratokiwa/Git/HSCSextansPMMeasurement")
from utils.utils import calR, per1_sigma, bts, err_all, errmag, intri_fix

def calculate_errs(data, config_dict, min_count=3):
        tick, mmin, mmax = config_dict["process"]["magtick"], config_dict["data"]["i_min"], config_dict["data"]["i_max"]
        bootstraps = config_dict["process"]["bootstraps"]
        pmracol, pmdeccol, magcol = config_dict["data"]["pmracol"], config_dict["data"]["pmdeccol"], config_dict["data"]["magcol"]

        imags = []
        im = np.arange(mmin, mmax, tick)
        counts = np.histogram(data[magcol], bins=im)[0]

        for imag_low, imag_high in zip(im[:-1], im[1:]):
                imags.append(data[(data[magcol] > imag_low) & (data[magcol] < imag_high)])
        imags = [tmp for tmp in imags if len(tmp) > min_count]

        pmraerr = [per1_sigma(tmp[pmracol]) for tmp in imags]
        pmdecerr = [per1_sigma(tmp[pmdeccol]) for tmp in imags]

        pmrabts = [np.std(np.apply_along_axis(per1_sigma, 1, bts(tmp[pmracol], bootstraps))) for tmp in imags]
        pmdecbts = [np.std(np.apply_along_axis(per1_sigma, 1, bts(tmp[pmdeccol], bootstraps))) for tmp in imags]

        errs = pd.DataFrame(list(zip(im[:-1][counts > min_count], counts[counts > min_count], pmraerr, pmdecerr, pmrabts, pmdecbts)), 
                        columns=["imag", "num", "pmraerr", "pmdecerr", "pmrabts", "pmdecbts"])
        return errs

def fit_curve_and_apply(df, x_col, y_col, sigma_col, p0=[20, 0.3, 20], bounds=((0, 0, 0), (np.inf, np.inf, np.inf))):
    popt, pcov = curve_fit(err_all, df[x_col], df[y_col], p0=p0, bounds=bounds, sigma=df[sigma_col])
    return popt

def plot_err(errs, errsq, popt_ra, popt_dec, img_path):
    fig = plt.figure(figsize=(7, 7))
    ax0 = fig.add_axes([0, 0.5, 1, 0.5])
    ax0.errorbar(errs.imag,
                errs.pmraerr,
                yerr=errs.pmrabts,
                capsize=6,
                fmt='o',
                markersize=8,
                color='tab:blue',alpha=0.6)

    ax0.plot(errs.imag,
            err_all(errs.imag, popt_ra[0], popt_ra[1], popt_ra[2]),
            c="tab:orange",
            label="PM dispersion",alpha=0.8,lw=3)
    ax0.plot(errs.imag,
            intri_fix(errs.imag, popt_ra[2]),linestyle="-.",
            c="tab:green",
            label="intrinsic dispersion",alpha=0.8,lw=3)

    ax0.plot(errs.imag,
            errmag(errs.imag, popt_ra[0], popt_ra[1]),linestyle="--",
            c="tab:red",
            label="photometric error",alpha=0.8,lw=3)

    ax0.set_ylabel(r"$\sigma_{\mu_\alpha}$"+' [mas/yr]', fontsize=20)

    ax0.set_ylim(-0.50, 14)
    ax0.set_xticklabels([])

    ax0.errorbar(errsq.imag,errsq.pmraerr,yerr=errsq.pmrabts, capsize=6,markersize=8,label="quasar",fmt='^',color="tab:pink",alpha=0.6)
    ax1 = fig.add_axes([0, 0, 1, 0.5])
    ax1.errorbar(errs.imag,
                errs.pmdecerr,
                yerr=errs.pmdecbts,
                capsize=6,
                fmt='o',
                markersize=8,
                color='tab:blue',alpha=0.6)

    ax1.plot(errs.imag,
            err_all(errs.imag, popt_dec[0], popt_dec[1], popt_dec[2]),
            c="tab:orange",
            label="PM variance",alpha=0.8,lw=3)
    ax1.plot(errs.imag,
            intri_fix(errs.imag, popt_dec[2]),linestyle="-.",
            c="tab:green",
            label="intrinsic dispersion",alpha=0.8,lw=3)

    ax1.plot(errs.imag,
            errmag(errs.imag, popt_dec[0], popt_dec[1]),linestyle="--",
            c="tab:red",
            label="photometric error",alpha=0.8,lw=3)
    
    ax1.errorbar(errsq.imag,errsq.pmdecerr,yerr=errsq.pmdecbts, capsize=6,markersize=8,label="quasar",fmt='^',color="tab:pink",alpha=0.6)

    ax1.set_xlabel(r'$\rm{i_{psf}}$' + ' [mag]', fontsize=20)
    ax1.set_ylim(-0.50, 14)
    ax1.set_ylabel(r"$\sigma_{\mu_\delta}$"+'  [mas/yr]', fontsize=20)

    ax0.legend(loc="upper center", fontsize=16)

    fig.savefig(img_path, bbox_inches="tight")

def main(data_path, quasar_path, output_path, img_path, config_path, params_path):
        ms_ccut= pd.read_csv(data_path)
        with open(config_path, "r") as f:
              config_dict = yaml.safe_load(f)
        center, r_tidal = config_dict["prefix"]["center"], config_dict["process"]["r_tidal"]
        racol, deccol = config_dict["data"]["racol"], config_dict["data"]["deccol"]
        magcol, pmraerrcol, pmdecerrcol = config_dict["data"]["magcol"], config_dict["data"]["pmraerrcol"], config_dict["data"]["pmdecerrcol"]
        ms_ccut["R"] = calR(ms_ccut[racol], ms_ccut[deccol], center)

        data = ms_ccut[ms_ccut.R > r_tidal].copy()
        errs = calculate_errs(data, config_dict, min_count=3)

        popt_ra = fit_curve_and_apply(errs, 'imag', 'pmraerr', 'pmrabts', p0=[20, 0.3, 20], bounds=((0, 0, 0), (np.inf, np.inf, np.inf)))
        popt_dec = fit_curve_and_apply(errs, 'imag', 'pmdecerr', 'pmdecbts', p0=[20, 0.3, 20], bounds=((0, 0, 0), (np.inf, np.inf, np.inf)))

        ms_ccut[pmraerrcol] = ms_ccut[magcol].apply(err_all, args=[popt_ra[0], popt_ra[1], popt_ra[2]])
        ms_ccut[pmdecerrcol] = ms_ccut[magcol].apply(err_all, args=[popt_dec[0], popt_dec[1], popt_dec[2]])
        #ms_ccut["pmra_estgalerr"] = ms_ccut.i_psfflux_mag.apply(errmag, args=[popt_ra[0], popt_ra[1]])
        #ms_ccut["pmdec_estgalerr"] = ms_ccut.i_psfflux_mag.apply(errmag, args=[popt_dec[0], popt_dec[1]])

        ms_ccut.to_csv(output_path, index=False)

        mq=pd.read_csv(quasar_path)
        errsq = calculate_errs(mq, config_dict, min_count=3)

        plot_err(errs, errsq, popt_ra, popt_dec, img_path)

        if os.path.exists(params_path):
                with open(params_path) as f:
                        params = json.load(f)
        else:
                params = {}

        params["errorest"] = {"pmra": popt_ra.tolist(), "pmdec": popt_dec.tolist()}
        with open(params_path, 'wt') as f:
                json.dump(params, f, indent=2, ensure_ascii=False)  
                
        return 0

if __name__ == '__main__':
        base_dir = "/Users/akiratokiwa/workspace/Sextans_final/"
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_path', type=str, default=f'{base_dir}catalog/product/starall_HSCS21a_PI_ccut.csv')
        parser.add_argument('--quasar_path', type=str, default=f'{base_dir}catalog/product/HSCS21a_SDSSQSO_pm.csv')
        parser.add_argument('--output_path', type=str, default=f'{base_dir}catalog/product/starall_HSCS21a_PI_errorest.csv')
        parser.add_argument('--img_path', type=str, default=f'{base_dir}img/plots/errorest.pdf')
        parser.add_argument('--config_path', type=str, default="/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/config.yaml")
        parser.add_argument('--params_path', type=str, default="/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/params.json")
        args = parser.parse_args()
        main(**vars(args))
