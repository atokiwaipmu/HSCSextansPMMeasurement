import pandas as pd

def main():
    base_dir = "/Users/akiratokiwa/workspace/Sextans_final/"
    catalog_dir = f"{base_dir}catalog/HSC-SSP/"

    m = pd.read_csv(f"{catalog_dir}HSC_S21a_-24_areakomy_wmjd.csv")
    m.drop("i_extendedness_value", axis=1, inplace=True)
    m_sm = pd.read_csv(f"{catalog_dir}brightstarmask_-24_S21a.csv")

    m_sm = m_sm[~m_sm[['i_mask_brightstar_halo', 'i_mask_brightstar_blooming', 'i_mask_brightstar_ghost15']].any(axis=1)]

    m = m.merge(m_sm[['# object_id', 'i_mask_brightstar_halo',
        'i_mask_brightstar_blooming', 'i_mask_brightstar_ghost15']], on="# object_id", how="inner")

    # Calculate the difference between 'i_cmodel_mag' and 'i_psfflux_mag'
    diff = m.i_cmodel_mag - m.i_psfflux_mag

    mg = m[(diff < -0.15) & (diff > -0.5)]
    ms = m[diff > -0.015]

    # Remove NaNs and save to CSV
    ms.dropna().to_csv(f"{base_dir}catalog/base/HSC_S21a_-24_areakomy_star.csv", index=False)
    mg.dropna().to_csv(f"{base_dir}catalog/base/HSC_S21a_-24_areakomy_galaxy.csv", index=False)

if __name__ == "__main__":
    main()