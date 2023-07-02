
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
import json
import yaml

def main():
    params_path = "/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/params.json"
    with open(params_path) as f:
        params = json.load(f)

    model = params["chosen_model"]
    ra, dec = params[model]["bestfit"][:2]
    pmra, pmdec = params["sxt_free"]["bestfit_1d"]

    config_path = "/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/config.yaml"
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    dist = config_dict["prefix"]["distance"]
    rv = config_dict["prefix"]["v_r"]
    r_sun, z_sun = config_dict["prefix"]["r_sun"], config_dict["prefix"]["z_sun"]
    v_sun = np.array(config_dict["prefix"]["v_sun"])

    icrs = SkyCoord(ra=ra*u.deg, dec=dec*u.deg,distance= dist *u.kpc,
                    pm_ra_cosdec=pmra*u.mas/u.yr, pm_dec=pmdec*u.mas/u.yr,
                    radial_velocity=rv*u.km/u.s, galcen_distance=r_sun*u.kpc,z_sun=z_sun*u.kpc, 
                    galcen_v_sun=v_sun*u.km/u.s)
    
    galcen = icrs.transform_to("galactocentric")
    
    params["galcen"] = {"v_x": galcen.v_x.value, "v_y": galcen.v_y.value, "v_z": galcen.v_z.value}

    with open(params_path, 'wt') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()