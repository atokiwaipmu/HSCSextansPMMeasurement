import pandas as pd
import numpy as np
import yaml

import sys
sys.path.append("/Users/akiratokiwa/Git/HSCSextansPMMeasurement/")
from utils.utils import Zs, apply_async_pool, cali_star_galerr, grid
from data.dataprocess import replace_inf_nan_dropna, compute_pm

def main():
    base_dir = "/Users/akiratokiwa/workspace/Sextans_final/"
    quaser_path = base_dir + "catalog/product/HSCS21a_SDSSQSO_PI.csv"
    gal_path = base_dir + "catalog/product/HSCS21a_PI_galaxy_cl.csv"
    output_path = base_dir + "catalog/product/HSCS21a_SDSSQSO_pm.csv"
    config_path = "/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/config.yaml"

    mq=pd.read_csv(quaser_path)
    mq = compute_pm(mq)    
    mg=pd.read_csv(gal_path)

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    xedges=np.arange(config_dict["data"]["xmin"],config_dict["data"]["xmax"],config_dict["data"]["gridtick"])
    yedges=np.arange(config_dict["data"]["ymin"],config_dict["data"]["ymax"],config_dict["data"]["gridtick"])

    data_mg = apply_async_pool(8, grid,yedges, mg, xedges, yedges)
    Znum_g, Zra_g, Zdec_g, Zrasem_g, Zdecsem_g = Zs(data_mg, "pmra", "pmdec")

    mq_tmp=pd.concat(apply_async_pool(8, cali_star_galerr, yedges,
                                    mq, xedges, yedges, Zra_g, Zdec_g, Zrasem_g, Zdecsem_g))
    
    col_list = ["pmra_cl", "pmdec_cl", "pmra_galerr", "pmdec_galerr", "i_psfflux_mag"]
    mq_tmp = replace_inf_nan_dropna(mq_tmp, col_list)

    # Write to csv
    mq_tmp.to_csv(output_path, index=False)

    return 0

if __name__ == '__main__':
    main()
