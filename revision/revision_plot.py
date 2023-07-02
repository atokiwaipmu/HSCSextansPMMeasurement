

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.optimize import curve_fit
from scipy import interpolate
from utils import medians, calR
import matplotlib.patches as patches

center = [153.2620833, -1.6147221]

def expo(x,c0,c1):
    return np.exp(c0+c1*x)


def plot_galcon(ms_SSP, mg_SSP, CMD_func, giwidth, imgpath):
    fig,ax = plt.subplots(1,2,figsize=(14,7),tight_layout=True)
    z = ax[0].hist(ms_SSP.i_psfflux_mag, histtype="step",range=(16,24),bins=80,label="HSC-SSP")
    overdense = z[0][z[1][:-1]>22.5]
    ax[0].hlines(np.min(overdense),16,24,color="tab:orange",linestyle="--")
    overdense_percentage = np.sum(overdense-np.min(overdense))/np.sum(overdense)
    ax[0].fill_between(z[1][z[1]>22.5][1:],np.min(overdense),overdense,color="tab:orange",alpha=0.5, label="overdense region: {:.1f}%".format(overdense_percentage*100))
    ax[0].fill_between(z[1][z[1]>22.5][1:],0,np.min(overdense),color="tab:blue",alpha=0.5)
    ax[0].legend(loc="upper left",fontsize=16, frameon=True, edgecolor="black", facecolor="white", framealpha=1)

    xedges = np.arange(-1.5,2.04, 0.02)
    yedges = np.arange(22.05, 24.54, 0.02)
    norms = colors.LogNorm(vmin=1, vmax=100)
    xL = np.linspace(22.05,24.54,100)

    hist0 = np.histogram2d(ms_SSP.gc-ms_SSP.ic, ms_SSP.ic,bins=(xedges, yedges))
    z0 = hist0[0].T
    mappable0 = ax[1].pcolormesh(xedges, yedges, z0, cmap="hot",norm=norms, alpha=0.8)
    indexes = np.random.randint(len(mg_SSP), size=10**4)
    n_mgccut = len(mg_SSP[(mg_SSP.i_psfflux_mag>22.5)&(mg_SSP.gi<CMD_func(mg_SSP.i_psfflux_mag)+giwidth(mg_SSP.i_psfflux_mag))&(mg_SSP.gi>CMD_func(mg_SSP.i_psfflux_mag)-giwidth(mg_SSP.i_psfflux_mag))].copy())
    inside_colorcut= n_mgccut/len(mg_SSP[mg_SSP.i_psfflux_mag>22.5].copy())
    plt.scatter(mg_SSP.iloc[indexes].gc-mg_SSP.iloc[indexes].ic, mg_SSP.iloc[indexes].ic,color="tab:blue",s=2, alpha=0.6, label="inside colorcut: {:.1f}%".format(inside_colorcut*100))
    ax[1].plot(CMD_func(xL)+giwidth(xL), xL,c="black",linestyle="-",lw=3)
    ax[1].plot(CMD_func(xL)-giwidth(xL), xL,c="black",linestyle="-",lw=3)
    ax[1].hlines(22.5,-1.5,2.04,color="black",linestyle="--", lw=3)
    ax[1].set_xlabel(r'$\rm{(g - i)_{0}[mag]}$')
    ax[1].set_ylabel(r'$\rm{i_{0, PSF}[mag]}$')
    ax[1].axis([-1.55, 2.05, 24.05, 22.05])
    ax[1].legend(loc="upper left",fontsize=16, markerscale=5, frameon=True, edgecolor="black", facecolor="white", framealpha=1)
    fig.savefig(imgpath, bbox_inches='tight')

def plot_maghist(ms_ccut, area, outer_area, imgpath):
    fig = plt.figure(figsize=(7, 4))

    ax0 = fig.add_axes([0, 0, 1, 1])

    tmp = ms_ccut[calR(ms_ccut.i_sdsscentroid_ra, ms_ccut.i_sdsscentroid_dec, center=center) < 0.5].copy()
    ax0.hist(tmp.i_psfflux_mag, bins=60, range=(18,24),color="tab:orange"
            ,  label=r"$R < 0.5^\circ$"+"(Sum = {0})".format(len(tmp)), histtype="step")

    tmp = ms_ccut[calR(ms_ccut.i_sdsscentroid_ra, ms_ccut.i_sdsscentroid_dec, center=[154, 0.5]) < 0.5].copy()

    tmp = ms_ccut[calR(ms_ccut.i_sdsscentroid_ra, ms_ccut.i_sdsscentroid_dec, center=center) > 1.5].copy()
    ax0.hist(tmp.i_psfflux_mag, bins=60, range=(18,24),color="tab:green"
            ,  label=r"$R > 1.5^\circ$"+"(Sum = {0})".format(len(tmp)), histtype="step")

    counts, bins = np.histogram(tmp.i_psfflux_mag, bins=60, range=(18,24))
    counts = counts*area/outer_area
    ax0.stairs(counts, bins,color="tab:blue"
            , label=r"same area as $R < 0.5^\circ$"+"(Sum = {0:.1f})".format(np.sum(counts)))

    ax0.set_ylabel('Number', fontsize=16)
    ax0.set_xlabel(r'$i_\mathrm{PSF}$ [mag]')
    ax0.set_yscale("log")
    ax0.set_xlim(18,24)
    ax0.legend(loc="upper left", fontsize=12)

    fig.savefig(imgpath, bbox_inches='tight')

def plot_dist(ms_err, centers, eps, rp, theta, imgpath):
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.scatter(ms_err.i_sdsscentroid_ra,
            ms_err.i_sdsscentroid_dec,
            c="black",
            s=1,
            alpha=0.2)

    ax.scatter(153.2620833, -1.6147221, s=10, c="tab:blue")
    ax.set_xlabel('RA [deg]', fontsize=16)
    ax.set_ylabel('Dec [deg]', fontsize=16)
    ax.set_ylim(-2.5, 1.7)

    c = patches.Ellipse(xy=centers,
                        width=(1 - eps) *rp* 4,
                        height=rp * 4,
                        angle=-theta * 180 / np.pi,
                        fill=False,
                        color="tab:orange",linestyle="-",
                        lw=2,label=r"$R_e = 2 \times R_h$")
    ax.add_patch(c)

    c2 = patches.Circle(xy=center,
                        radius=1.5,
                        fill=False,
                        color="tab:red",
                        linestyle="--",
                        lw=2,label=r"R = 1.5$^\circ$")
    ax.add_patch(c2)

    c3 = patches.Circle(xy=center,
                        radius=0.5,
                        fill=False,
                        color="tab:red",
                        linestyle="-",
                        lw=2,label=r"R = 0.5$^\circ$")
    ax.add_patch(c3)
    """
    c4 = patches.Circle(xy=[154, 0.5],
                        radius=0.5,
                        fill=False,
                        color="tab:blue",
                        linestyle="-",
                        lw=2,label="comparison")
    ax.add_patch(c4)
    """
    ax.legend(loc="upper left", fontsize=16)

    fig.savefig(imgpath,bbox_inches="tight")

def main():
    ms_SSP=pd.read_csv("/Users/akiratokiwa/workspace/Sextans_final/catalog/base/HSC_S21a_-24_areakomy_star.csv")
    mg_SSP=pd.read_csv("/Users/akiratokiwa/workspace/Sextans_final/catalog/base/HSC_S21a_-24_areakomy_galaxy.csv")
    m_ext=pd.read_csv('/Users/akiratokiwa/workspace/Sextans_final/catalog/HSC-SSP/HSCS21A_ext.csv')
    ms_SSP = ms_SSP.merge(m_ext, on="# object_id").copy()
    mg_SSP = mg_SSP.merge(m_ext, on="# object_id").copy()
    ms_SSP["gc"] = ms_SSP.g_psfflux_mag - ms_SSP.a_g
    ms_SSP["rc"] = ms_SSP.r_psfflux_mag - ms_SSP.a_r
    ms_SSP["ic"] = ms_SSP.i_psfflux_mag - ms_SSP.a_i

    mg_SSP["gc"] = mg_SSP.g_psfflux_mag - mg_SSP.a_g
    mg_SSP["rc"] = mg_SSP.r_psfflux_mag - mg_SSP.a_r
    mg_SSP["ic"] = mg_SSP.i_psfflux_mag - mg_SSP.a_i

    mg_SSP["gi"] = mg_SSP.gc - mg_SSP.ic   

    #CMD_funs
    base_dir = "/Users/akiratokiwa/workspace/Sextans_final/"
    ms2=pd.read_csv(base_dir+'catalog/product/starall_HSCS21a_PI.csv')
    
    ms2 = ms2[ms2["# object_id"] != 42098376082216514]
    ms2.reset_index(drop=True, inplace=True)
    ms2["gierr"] = np.sqrt(ms2.g_psfflux_magerr**2 + ms2.i_psfflux_magerr**2)
    ms_core = ms2[((ms2.i_sdsscentroid_ra - center[0])**2 + (ms2.i_sdsscentroid_dec - center[1])**2) < 0.5**2]

    SGBRGB = ms_core[(ms_core.gi < 1) & (ms_core.gi > 0.1) & ((ms_core.gi > 0.7) | (ms_core.ic > 22) | (-3/30*(ms_core.ic-19)+0.7 < ms_core.gi))]
    
    xmin, xmax, n_tmp = 18, 24, 30
    xL = np.linspace(xmin, xmax, n_tmp)
    md_gi, errs_gi = medians(SGBRGB, SGBRGB.ic, SGBRGB.gi, xmin, xmax, n_tmp, 0)
    md_gierr, errs_gierr = medians(SGBRGB, SGBRGB.ic, SGBRGB.gierr, xmin, xmax, n_tmp, 0)

    popt1, pcov1 = curve_fit(expo, xL[10:],md_gierr[10:], sigma=errs_gierr[10:])
    popt2, pcov2 = curve_fit(expo, xL[10:],errs_gi[10:])

    CMD_func = interpolate.CubicSpline(xL,md_gi)

    def giwidth(ipsf, popt1=popt1, popt2=popt2):
        return np.sqrt(expo(ipsf,popt2[0], popt2[1])**2-expo(ipsf,popt1[0], popt1[1])**2)+3*expo(ipsf,popt1[0], popt1[1])
    
    plot_galcon(ms_SSP, mg_SSP, CMD_func, giwidth, "/Users/akiratokiwa/workspace/Sextans_final/img/plots/overdense.pdf")

    ms_ccut= pd.read_csv(base_dir+'catalog/product/starall_HSCS21a_PI_ccut.csv')
    ZdistR = np.loadtxt(base_dir+"params_estimated/ZdistR.txt")
    area = len(ZdistR[(ZdistR > 0)&(ZdistR < 0.5)]) * (0.025**2)
    outer_area = len(ZdistR[(ZdistR > 1.5)]) * (0.025**2)
    plot_maghist(ms_ccut, area, outer_area, "/Users/akiratokiwa/workspace/Sextans_final/img/plots/maghist.pdf")

    params = np.load(base_dir + "params_estimated/params.npy", allow_pickle=True).item()
    centers, eps, theta, rp = params["centers"], params["eps"], params["theta"], params["rp"]

    ms_err = pd.read_csv(base_dir+'catalog/product/starall_HSCS21a_PI_errorest.csv')
    plot_dist(ms_err, centers, eps, rp, theta,  "/Users/akiratokiwa/workspace/Sextans_final/img/plots/dist.png")

if __name__ == "__main__":
    main()