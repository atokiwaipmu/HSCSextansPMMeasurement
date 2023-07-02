
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import calR
from structure.structure import run_mcmc, loglikelihood

center = [153.2620833, -1.6147221]

def main():
    base_dir = "/Users/akiratokiwa/workspace/Sextans_final/"
    ms_ccut = pd.read_csv(base_dir+'catalog/product/starall_HSCS21a_PI_ccut.csv')
    
    b=10**4

    ZdistR = np.loadtxt(base_dir+"params_estimated/ZdistR.txt")
    area = len(ZdistR[(ZdistR > 0)&(ZdistR < 1.5)]) * (0.025**2)
    outer_area = len(ZdistR[(ZdistR > 1.5)]) * (0.025**2)
    outer_nd = len(ms_ccut[calR(ms_ccut.i_sdsscentroid_ra, ms_ccut.i_sdsscentroid_dec, center=center) > 1.5]) / outer_area
    
    prefix = [area,outer_nd, b]

    initp_dict = {"plum": [153.26, -1.614, 57, 0.2, 20], 
              "king": [153.26, -1.614, 57, 0.2, 20, 90], 
              "expo": [153.26, -1.614, 57, 0.2, 20]}
    initp_rand_dict = {"plum": [0.1, 0.1, 10, 0.1, 5], 
                    "king": [0.1, 0.1, 10, 0.1, 5, 10], 
                    "expo": [0.1, 0.1, 10, 0.1, 5]}

    series = {"plum":{"bestfit":[], "err":[], "likelihood":[]}, "king":{"bestfit":[], "err":[], "likelihood":[]}, "expo":{"bestfit":[], "err":[], "likelihood":[]}}
    for max_r in np.arange(0.5, 3, 0.2):
        print("max_r = {}".format(max_r))
        data  = ms_ccut[calR(ms_ccut.i_sdsscentroid_ra, ms_ccut.i_sdsscentroid_dec, center=center) < max_r].copy()
        inputs = [data.i_sdsscentroid_ra,data.i_sdsscentroid_dec,len(data)]
        for mcmc_type in ["plum", "king", "expo"]:
            initp = np.array(initp_dict[mcmc_type])
            initp_rand = np.array(initp_rand_dict[mcmc_type])
            samples = run_mcmc(initp, initp_rand, inputs, prefix, mcmc_type, max_n=3000)
            series[mcmc_type]["bestfit"].append(np.median(samples, axis=0))
            series[mcmc_type]["err"].append(np.std(samples, axis=0))
            series[mcmc_type]["likelihood"].append(loglikelihood(np.median(samples, axis=0), inputs, prefix, model=mcmc_type)/inputs[2])

    """
    series_bgerr = {"plum":{"bestfit":[], "err":[]}, "king":{"bestfit":[], "err":[]}, "expo":{"bestfit":[], "err":[]}}
    data  = ms_ccut[calR(ms_ccut.i_sdsscentroid_ra, ms_ccut.i_sdsscentroid_dec, center=center) < 1.5].copy()
    inputs = [data.i_sdsscentroid_ra,data.i_sdsscentroid_dec,len(data)]
    np.random.seed(10)
    for touter_nd in outer_nd+0.2*outer_nd*np.random.normal(size=100):
        print("outer_nd = {}".format(touter_nd))
        prefix = [area,touter_nd, b]
        for mcmc_type in ["plum", "king", "expo"]:
            initp = np.array(initp_dict[mcmc_type])
            initp_rand = np.array(initp_rand_dict[mcmc_type])
            samples = run_mcmc(initp, initp_rand, inputs, prefix, mcmc_type, max_n=3000)
            series_bgerr[mcmc_type]["bestfit"].append(np.median(samples, axis=0))
            series_bgerr[mcmc_type]["err"].append(np.std(samples, axis=0))
    """

    # save
    np.save(base_dir+"params_estimated/series_all.npy", series)
    #np.save(base_dir+"params_estimated/series_bgerr.npy", series_bgerr)

if __name__ == "__main__":
    main()

