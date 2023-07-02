
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import sys
import argparse
from utils.utils import flux_to_mag, rad_to_deg, fluxerr_to_magerr

def set_hist2d(ax, data, title, xedges, yedges, norm):
    hist, *_ = np.histogram2d(data.i_ra, data.i_dec, bins=(xedges, yedges))
    z = hist.T
    ax.pcolormesh(xedges, yedges, z, cmap="hot", norm=norm)
    ax.set_title(title, fontsize=32)
    ax.set_xlabel(r'$\rm{RA[deg]}$', fontsize=32)
    ax.set_ylabel(r'$\rm{Dec[deg]}$', fontsize=32)
    ax.set_aspect('equal')

def set_magnitude_histogram(ax, datasets, labels, bins):
    for data, label in zip(datasets, labels):
        ax.hist(data.i_psfflux, histtype="step", bins=bins, label=label)
    ax.set_yscale('log')
    ax.set_title("Magnitude Histogram", fontsize=32)
    ax.set_xlabel(r'$\rm{i_{PSF}[mag]}$', fontsize=32)
    ax.set_ylabel(r'$\rm{Number\ of\ objects}$', fontsize=32)
    ax.legend(loc="upper left")

def plot_PI(m_all, ms, mg, img_path):
    xedges = np.arange(149.75, 156.80, 0.05)
    yedges = np.arange(-4.90, 1.70, 0.05)
    fig, ax = plt.subplots(2, 2, figsize=(28, 20), tight_layout=True)
    
    norm = colors.Normalize(vmin=0, vmax=500)
    set_hist2d(ax[0, 0], m_all, "All objects", xedges, yedges, norm)

    set_magnitude_histogram(ax[0, 1], [m_all, ms, mg], ["All objects", "Star", "Galaxy"], 100)

    norm2 = colors.Normalize(vmin=0, vmax=200)
    set_hist2d(ax[1, 0], ms, "Star", xedges, yedges, norm2)
    set_hist2d(ax[1, 1], mg, "Galaxy", xedges, yedges, norm2)

    fig.savefig(img_path, bbox_inches='tight', dpi=300)

def apply_transformations(df, transformations):
    for col, func in transformations:
        print(col)
        df[col] = df[col].apply(func)
    return df

def process_band(df, band, mjd_values):
    print(band)
    col_band = [f'{band}_'+col for col in ['ra', 'dec', 'psfflux', 'cmodel', 'ra_err', 'dec_err', 'psfflux_err']]
    col_band.insert(0, 'id')
    df_band = df.dropna(subset=col_band[:-1])[col_band].copy()
    df_band["mjd"] = mjd_values[band]
    if band=="i2":
        df_band.columns = df_band.columns.str.replace('^i2_', 'i_')
    print(df_band.shape)
    return df_band

def main(data_path, mid_dir, output_dir, img_path, flag=False):
    bands = ["i", "i2", "g"]
    mjd_values = {'i': 56980, 'i2': 57481, 'g': 56981}
    # Define a list of tuples (column_name, function) to apply
    transformations = [('i_ra', rad_to_deg), ('i_dec', rad_to_deg), 
                    ('i2_ra', rad_to_deg), ('i2_dec', rad_to_deg),
                    ('i_psfflux', flux_to_mag), ('i_cmodel', flux_to_mag),
                    ('i2_psfflux', flux_to_mag), ('i2_cmodel', flux_to_mag),
                    ('g_ra', rad_to_deg), ('g_dec', rad_to_deg),
                    ('g_psfflux', flux_to_mag), ('g_cmodel', flux_to_mag)]

    file_trans = mid_dir + 'HSCPI_fits_trans.csv'
    file_i = mid_dir + 'HSCPI_i.csv'
    file_ii = mid_dir + 'HSCPI_i+i2.csv'
    file_concat = mid_dir + 'HSCPI_concat.csv'

    if (not os.path.exists(file_trans)) or flag:
        m_komy=pd.read_csv(data_path)
        m = m_komy[m_komy.is_primary].copy()
        print("m.shape", m.shape)
        m = apply_transformations(m, transformations)
        m.to_csv(file_trans,index=False)
    m = pd.read_csv(file_trans)

    df_tmp = {}
    if (not os.path.exists(file_i)) or flag:
        for band in bands:
            df_tmp[band] = process_band(m, band, mjd_values)
            df_tmp[band].to_csv(mid_dir + f"HSCPI_{band}.csv",index=False)
    else:
        for band in bands:
            df_tmp[band] = pd.read_csv(mid_dir + f"HSCPI_{band}.csv")
            print(df_tmp[band].shape)

    if (not os.path.exists(file_ii)) or flag:
        m_tmp = pd.concat([df_tmp["i"], df_tmp["i2"]])
        m_tmp = m_tmp[~m_tmp.duplicated(subset=["id"],keep="last")]
        m_tmp = m_tmp.merge(df_tmp["g"], on="id", how="inner")
        m_all = m_tmp[m_tmp.i_psfflux < 24.5]
        print(m_all.shape)
        m_tmp.to_csv(file_ii, index=False)
        m_all.to_csv(file_concat, index=False)
    else:
        m_all = pd.read_csv(file_concat)
        print(m_all.shape)

    mg = m_all[(m_all.i_cmodel - m_all.i_psfflux < -0.15) & (m_all.i_cmodel - m_all.i_psfflux > -0.5)]
    ms = m_all[m_all.i_cmodel - m_all.i_psfflux > -0.015]
    print(mg.shape, ms.shape)

    ms.to_csv(output_dir + "HSCPI_star.csv", index=False)
    mg.to_csv(output_dir + "HSCPI_galaxy.csv", index=False)

    plot_PI(m_all, ms, mg, img_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/Users/akiratokiwa/workspace/Sextans_final/catalog/HSC-PI/yasuda_fits_merged.csv")
    parser.add_argument("--mid_dir", type=str, default="/Users/akiratokiwa/workspace/Sextans_final/catalog/HSC-PI/")
    parser.add_argument("--output_dir", type=str, default="/Users/akiratokiwa/workspace/Sextans_final/catalog/base/")
    parser.add_argument("--img_path", type=str, default="/Users/akiratokiwa/workspace/Sextans_final/img/plots/HSCPI.png")
    parser.add_argument("--flag", type=bool, default=False)
    args = parser.parse_args()
    main(args.data_path, args.mid_dir, args.output_dir, args.img_path, args.flag)