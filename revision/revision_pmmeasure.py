
import numpy as np
import pandas as pd
from utils import calRe
from scripts.pmmeasurement import run_mcmc, chi2_1d
import bisect
import scipy.interpolate as interpolate


def main():
    base_dir = "/Users/akiratokiwa/workspace/Sextans_final/"
    params = np.load(base_dir + "params_estimated/params.npy", allow_pickle=True).item()
    rp, rt = params["rp"], params["rt"]

    radials = np.load(base_dir + "params_estimated/numberdensity.npy", allow_pickle=True)
    rs, r_all, r_pmra, r_pmraerr, r_pmdec, r_pmdecerr, r_numd, r_numderr, n_MW, r_area = radials
    r_num = r_numd * r_area

    n_2core = bisect.bisect(r_all,2*rp)
    n_end = len(r_num)

    n_end_dict = {"sxt_free":n_2core}

    initp_dict = {"sxt_free":[-0.409,0.083]}
    
    initp_rand_dict = {"sxt_free":[0.1,0.1]}

    samples_1d = np.load(base_dir + "params_estimated/samples_1d.npy", allow_pickle=True).item()
    bg = np.median(samples_1d["sxt_MW_free"], axis=0)[2:4]
    bgerr = np.std(samples_1d["sxt_MW_free"], axis=0)[2:4]
    print(bg, bgerr)
    bg_list = [bg[0]*(1+0.2*np.random.normal(size=100)), bg[1]*(1+0.2*np.random.normal(size=100))]
    bg_list = np.array(bg_list).T

    series_bgerr = {"bestfit":[], "err":[]}
    np.random.seed(10)
    for tbg_ra, tbg_dec in bg_list:
        print("bg_ra = {}, bg_dec = {}".format(tbg_ra, tbg_dec))
        initp = np.array(initp_dict["sxt_free"])
        initp_rand = np.array(initp_rand_dict["sxt_free"])
        n_end = n_end_dict["sxt_free"]
        obs = [r_num[1:n_end], r_pmra[1:n_end], r_pmdec[1:n_end], r_pmraerr[1:n_end],
                r_pmdecerr[1:n_end],np.array(n_MW)[1:n_end]]
        bgr = [tbg_ra, tbg_dec]
        samples = run_mcmc(initp, initp_rand, chi2_1d, obs, bgr)
        series_bgerr["bestfit"].append(np.median(samples, axis=0))
        series_bgerr["err"].append(np.std(samples, axis=0))

    np.save(base_dir+"params_estimated/pmmeasure_bgerr.npy", series_bgerr)

if __name__ == "__main__":
    main()