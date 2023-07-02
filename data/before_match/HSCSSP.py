import pandas as pd
import os
import sys
import argparse

def main(data_path, mask_path, output_dir):
    # Read CSV files
    if not os.path.exists(data_path):
        print("Data File not found.")
        sys.exit()
    if not os.path.exists(mask_path):
        print("Mask File not found.")
        sys.exit()
    if not os.path.exists(output_dir):
        print("Output Directory not found.")
        sys.exit()
    m = pd.read_csv(data_path)
    m_sm = pd.read_csv(mask_path)

    # remove unnecessary columns
    m.drop("i_extendedness_value", axis=1, inplace=True)

    # Remove bright stars
    mask_cols = ['i_mask_brightstar_halo', 'i_mask_brightstar_blooming', 'i_mask_brightstar_ghost15']
    m_sm = m_sm[~m_sm[mask_cols].any(axis=1)]
    m = m.merge(m_sm[['# object_id'].extend(mask_cols)], on="# object_id", how="inner")

    # Calculate the difference between 'i_cmodel_mag' and 'i_psfflux_mag'
    diff = m.i_cmodel_mag - m.i_psfflux_mag

    mg = m[(diff < -0.15) & (diff > -0.5)]
    ms = m[diff > -0.015]

    # Remove NaNs and save to CSV
    ms.dropna().to_csv(f"{output_dir}HSC_S21a_-24_areakomy_star.csv", index=False)
    mg.dropna().to_csv(f"{output_dir}HSC_S21a_-24_areakomy_galaxy.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/Users/akiratokiwa/workspace/Sextans_final/catalog/HSC-SSP/HSC_S21a_-24_areakomy_wmjd.csv")
    parser.add_argument("--mask_path", type=str, default="/Users/akiratokiwa/workspace/Sextans_final/catalog/HSC-SSP/brightstarmask_-24_S21a.csv")
    parser.add_argument("--output_dir", type=str, default="/Users/akiratokiwa/workspace/Sextans_final/catalog/base/")
    args = parser.parse_args()
    main(args.data_path, args.mask_path, args.output_dir)