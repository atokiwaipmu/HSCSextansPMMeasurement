"""
This script performs plotting of the results.

Author: Akira Tokiwa
Refined by: OpenAI's ChatGPT
"""

import numpy as np
import matplotlib.pyplot as plt
import corner
import matplotlib.colors as colors
import bisect
from scipy.optimize import curve_fit
import json
import sys
sys.path.append("/Users/akiratokiwa/Git/HSCSextansPMMeasurement/")
from utils.utils import proplummer, proking, proexpo

def plot_wedge(flat_samples, labels, img_path):
    """
    Plots a wedge plot using the corner library.

    Args:
        flat_samples (np.array): The flattened MCMC samples.
        labels (list): List of labels for the plot.
        img_path (str): Path to save the plot.

    Returns:
        None.
    """
    fig = plt.figure(figsize=(7,7))
    corner.corner(flat_samples, labels=labels,
                           quantiles=[0.16, 0.5, 0.84],title_fmt= '.3f',
                           show_titles=True, title_kwargs={"fontsize": 9,"pad" : 6},fig=fig)
    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=8)

    fig.savefig(img_path, bbox_inches='tight', dpi=300)


def plot_rad(r_obs, bgr, popt, rads, errs, profits, bestfit_sxt,  img_path):
    """
    Plots the radial profile of the proper motions and number density.

    Args:
        r_obs (np.array): The radial data.
        bgr (np.array): The background parameters.
        popt (list): The optimized parameters for the profile fits.
        rads (np.array): The radii.
        errs (np.array): The errors.
        profits (list): The profile fit functions.
        bestfit_sxt (np.array): The best fit parameters for the Sextans.
        img_path (str): Path to save the plot.

    Returns:
        None.
    """
    # Set up the figure and axes
    fig = plt.figure(figsize=(7, 10))
    ax0 = fig.add_axes([0, 0.33, 1, 0.33])
    rs, r_all, r_pmra, r_pmraerr, r_pmdec, r_pmdecerr, r_numd, r_numderr, n_MW, r_area = r_obs
    poptp, poptk, popte = popt
    rp, re, rc, rt = rads
    err_sxt, err_sxtMW = errs
    pfit, kfit, efit = profits

    # Plot the data
    ax0.errorbar(r_all,
                r_pmra,
                yerr=r_pmraerr,
                capsize=2,
                fmt='o',
                markersize=5,
                color='black')

    ax1 = fig.add_axes([0, 0, 1, 0.33])
    ax1.errorbar(r_all,
                r_pmdec,
                yerr=r_pmdecerr,
                capsize=2,
                fmt='o',
                markersize=5,
                color='black')

    ax0.fill_between(
        r_all, (bgr[1]+err_sxtMW[2]) * n_MW + (bestfit_sxt[0] +  err_sxt[0]) * (1 - n_MW),
        (bgr[1]-err_sxtMW[2]) * n_MW + (bestfit_sxt[0] -  err_sxt[0]) * (1 - n_MW),
        alpha=0.5,
        color="tab:red",label=r"$\mu_{\alpha,\mathrm{sxt}}=$"+"{0:0.3f}".format(bestfit_sxt[0]))
    ax1.fill_between(
        r_all, (bgr[2]+err_sxtMW[3]) * n_MW + (bestfit_sxt[1] +  err_sxt[1]) * (1 - n_MW),
        (bgr[2]-err_sxtMW[3]) * n_MW + (bestfit_sxt[1] -  err_sxt[1]) * (1 - n_MW),
        alpha=0.5,
        color="tab:red", label=r"$\mu_{\delta,\mathrm{sxt}}=$"+"{0:0.3f}".format(bestfit_sxt[1]))

    ax2 = fig.add_axes([0, 0.66, 1, 0.33])
    ax2.errorbar(r_all,
                r_numd,
                yerr=r_numderr,
                capsize=2,
                fmt='o',
                markersize=5,
                color='black')

    # Set the scales
    ax0.set_xscale("log")
    ax1.set_xscale("log")
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    tlimit = bisect.bisect(r_all,rt)
    ax2.plot(r_all,pfit(r_all,poptp[0]),color="tab:blue", label="Best-fit Plummer")
    ax2.plot(r_all[:tlimit],kfit(r_all[:tlimit],poptk[0]),color="tab:orange", label="Best-fit King")
    ax2.plot(r_all,efit(r_all,popte[0]),color="tab:green", label="Best-fit Exponential")
    ax2.vlines(2*rp, np.min(r_numd), np.max(r_numd),linestyle="--",color="red",
            label=r"$2R_h=$"+"{0:0.2f}[deg]".format(2*rp))

    ax0.hlines(bgr[1], 0.5, np.max(r_all),linestyle="--",color="blue",
            label=r"$\mu_{\alpha,\mathrm{MW}}=$"+"{0:0.3f}".format(bgr[1]))
    ax0.set_xlabel("elliptical radius" + r":$R_e$" + " [deg]")
    ax0.set_ylabel(r"$\mu_{\alpha}$" + " [mas/yr]")
    ax0.set_ylim(-3.5, 1)
    ax0.legend(loc="lower left",fontsize=16)

    ax1.hlines(bgr[2], 0.5, np.max(r_all),linestyle="--",color="blue",
            label=r"$\mu_{\delta,\mathrm{MW}}=$"+"{0:0.3f}".format(bgr[2]))
    ax1.set_xlabel("elliptical radius" + r":$R_e$" + " [deg]")
    ax1.set_ylabel(r"$\mu_{\delta}$" + " [mas/yr]")
    ax1.set_ylim(-4, 1)
    ax1.legend(loc="lower left",fontsize=16)

    ax2.set_ylabel(r"$\Sigma^{obs}$" + r" [/deg$^2$]")
    ax2.set_xticklabels([])


    ax2.hlines(bgr[0], 0.5, np.max(r_all),linestyle="--",color="blue",
            label=r"$\Sigma_b=$"+"{0:1.0f}".format(bgr[0])+r"[deg$^{-2}$]")
    ax2.legend(loc="lower left",fontsize=16)

    # Save the figure
    fig.savefig(img_path, bbox_inches="tight")

def plot_compare(bestfit_sxt, bestfit_sxt2d, err_sxt, err_sxt2d, img_path):
    """
    Plots a comparison of the proper motion results with previous works.

    Args:
        bestfit_sxt (np.array): The best fit parameters for the Sextans for 1D fit.
        bestfit_sxt2d (np.array): The best fit parameters for the Sextans for 2D fit.
        err_sxt (np.array): The errors for the Sextans for 1D fit.
        err_sxt2d (np.array): The errors for the Sextans for 2D fit.
        img_path (str): Path to save the plot.

    Returns:
        None.
    """
    # Set up the figure and axes
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_axes([0, 0, 1, 1])

    # Previous works
    ppmra = [-0.403,-0.373,-0.41,-0.40]
    ppmdec = [0.029,0.021,0.04,0.02]
    psigra = [0.021,0.019,0.01,0.01]
    psigdec = [0.021,0.02,0.01,0.01]
    pname = ["Li et al. (2021)","Martínez-García et al. (2021)","McConnachie & Venn (2020)", "Battaglia et al. (2022)"]
    pcolor = ["tab:grey",'tab:purple','tab:red', "tab:green"]

    # This work
    rpmra = [bestfit_sxt[0], bestfit_sxt2d[0]]
    rpmdec = [bestfit_sxt[1], bestfit_sxt2d[1]]
    rsigra = [err_sxt[0], err_sxt2d[0]]
    rsigdec = [err_sxt[1], err_sxt2d[1]]
    rname = [r"this work(1d, $\mu_\mathrm{MW}$ fixed)",r"this work(2d, $\mu_\mathrm{MW}$ fixed)"]
    rcolor = ['tab:orange','tab:blue']

    # Plot the data
    for i in range(len(ppmra)):  
        ax.errorbar(ppmra[i],
                    ppmdec[i],
                    xerr=psigra[i],
                    yerr=psigdec[i],
                    capsize=5,
                    fmt='o',
                    markersize=10,
                    color=pcolor[i],label=pname[i])

    for i in range(len(rpmra)):  
        ax.errorbar(rpmra[i],
                    rpmdec[i],
                    xerr=rsigra[i],
                    yerr=rsigdec[i],
                    capsize=5,
                    fmt='*',
                    markersize=16,
                    color=rcolor[i],label=rname[i])    

    # Set the labels
    ax.set_xlabel(r"$\mu_\alpha$",fontsize=16)
    ax.set_ylabel(r"$\mu_\delta$",fontsize=16)
    ax.set_aspect('equal')
    ax.legend(loc="lower center", fontsize=12, ncol=3,bbox_to_anchor=(0.43, -0.1,), borderaxespad=-6,)
    ax.tick_params(axis='both')

    # Save the figure
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1) 
    fig.savefig(img_path, bbox_inches="tight")

def main():
    """
    Main function to plot the results.

    Args:
        None.

    Returns:
        None.
    """
    # Set up the paths
    base_dir = "/Users/akiratokiwa/workspace/Sextans_final/"

    # Load the samples
    psamples = np.loadtxt(base_dir + "params_estimated/plum.txt")
    ksamples = np.loadtxt(base_dir + "params_estimated/king.txt")
    esamples = np.loadtxt(base_dir + "params_estimated/expo.txt")
    sample_1d = np.load(base_dir + "params_estimated/samples_1d.npy", allow_pickle=True)
    sample_1d_sxtMW = sample_1d.item().get("sxt_MW_free") 
    sample_1d_sxt = sample_1d.item().get("sxt_free")
    sample_2d = np.load(base_dir + "params_estimated/samples_2d.npy", allow_pickle=True)
    sample_2d_sxtMW = sample_2d.item().get("sxt_MW_free")
    sample_2d_sxt = sample_2d.item().get("sxt_free")

    # Set up the labels
    plabels = [r"$\alpha_0$", r"$\delta_0$", r"$\theta$", r"$\epsilon$", r"$r_{p}$"]
    klabels = [r"$\alpha_0$", r"$\delta_0$", r"$\theta$", r"$\epsilon$", r"$r_{c}$", r"$r_{t}$"]
    elabels = [r"$\alpha_0$", r"$\delta_0$", r"$\theta$", r"$\epsilon$", r"$r_{exp}$"]
    smlabels =[ r"$\mu^\mathrm{sxt}_\alpha$",r"$\mu^\mathrm{sxt}_\delta$",r"$\mu^\mathrm{MW}_\alpha$",r"$\mu^\mathrm{MW}_\delta$"]
    slabels = [ r"$\mu^\mathrm{sxt}_\alpha$",r"$\mu^\mathrm{sxt}_\delta$"]

    # Plot the wedges
    plot_wedge(psamples, plabels, base_dir + "img/plots/wedge_plum.png")
    plot_wedge(ksamples, klabels, base_dir + "img/plots/wedge_king.png")
    plot_wedge(esamples, elabels, base_dir + "img/plots/wedge_expo.png")
    plot_wedge(sample_1d_sxtMW, smlabels, base_dir + "img/plots/wedge_sxtMW1d.png")
    plot_wedge(sample_1d_sxt, slabels, base_dir + "img/plots/wedge_sxt1d.png")
    plot_wedge(sample_2d_sxtMW, smlabels, base_dir + "img/plots/wedge_sxtMW2d.png")
    plot_wedge(sample_2d_sxt, slabels, base_dir + "img/plots/wedge_sxt2d.png")

    # Load the radial data
    radials = np.load(base_dir + "params_estimated/numberdensity.npy", allow_pickle=True)
    rs, r_all, r_pmra, r_pmraerr, r_pmdec, r_pmdecerr, r_numd, r_numderr, n_MW, r_area = radials

    # Load the parameters
    params_path = "/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/params.json"
    with open(params_path) as f:
        params = json.load(f)

    # Set up the parameters
    outer_nd = params["outer_nd"]
    rp = params["plum"]["bestfit"][-1]/60
    re = params["expo"]["bestfit"][-1]/60
    rc, rt = params["king"]["bestfit"][-2]/60, params["king"]["bestfit"][-1]/60
    rads = rp, re, rc, rt

    bgs = params["sxt_MW_free"]["bestfit_1d"][2:4]
    bgr = np.hstack([outer_nd, bgs])
    bestfit_sxt = params["sxt_free"]["bestfit_1d"]
    bestfit_sxt2d = params["sxt_free"]["bestfit_2d"]
    err_sxt = params["sxt_free"]["bestfit_err_1d"]
    err_sxt2d = params["sxt_free"]["bestfit_err_2d"]
    err_sxtMW = params["sxt_MW_free"]["bestfit_err_1d"]
    errs = err_sxt, err_sxtMW

    # Define the profile fits
    def pfit(x,a):
        return a*proplummer(x,rp) + outer_nd  

    def kfit(x,a):
        return a*proking(x,rc, rt) + outer_nd  

    def efit(x,a):
        return a*proexpo(x,re) + outer_nd  
    profits = pfit, kfit, efit

    # Fit the profiles
    poptp,pcovp = curve_fit(pfit, r_all,r_numd, sigma=r_numderr)
    tlimit = bisect.bisect(r_all,rt)
    poptk,pcovk = curve_fit(kfit, r_all[:tlimit],r_numd[:tlimit], sigma=r_numderr[:tlimit])
    popte,pcove = curve_fit(efit, r_all,r_numd, sigma=r_numderr)
    popt = [poptp, poptk, popte]

    # Plot the radial profile
    img_path = base_dir + "img/plots/pm_radial_profile.pdf"
    plot_rad(radials, bgr, popt, rads, errs, profits, bestfit_sxt,  img_path)

    # Plot the comparison
    img_path = base_dir + "img/plots/pm_compare.pdf"
    plot_compare(bestfit_sxt, bestfit_sxt2d, err_sxt, err_sxt2d, img_path)

if __name__ == '__main__':
    main()