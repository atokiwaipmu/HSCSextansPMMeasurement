"""
This script performs computations related to the half-light radius for different models of celestial bodies.

Author: Akira Tokiwa
"""

import json
import numpy as np
from scipy.optimize import fsolve


def integrate_sigma_king(t: float, a: float) -> float:
    """
    Integral of SigmaKing.

    Args:
        t: The variable of integration.
        a: The model parameter.

    Returns:
        The value of the integral at t.
    """
    return (0.5 * ((t**2 - 4 * np.sqrt(1 + a**2) * np.sqrt(1 + t**2)) / (1 + a**2) + np.log(1 + t**2)))


def compute_half_light_radius(R_c: float, R_t: float) -> float:
    """
    Compute the half-light radius.

    Args:
        R_c: The core radius.
        R_t: The tidal radius.

    Returns:
        The half-light radius.
    """
    a = R_t / R_c
    equation = lambda xh: integrate_sigma_king(xh, a) - 0.5 * (integrate_sigma_king(a, a) + integrate_sigma_king(0, a))
    xh_initial_guess = 20
    xh_solution = fsolve(equation, xh_initial_guess)
    return xh_solution[0] * R_c


def estimate_error_in_half_light_radius(rc: float, rt: float, rcerr: float, rterr: float) -> float:
    """
    Estimate the error in the half-light radius using Monte Carlo simulations.

    Args:
        rc: The value of R_c.
        rt: The value of R_t.
        rcerr: The error in R_c.
        rterr: The error in R_t.

    Returns:
        The estimated error in the half-light radius.
    """
    # Number of simulations
    N = 100

    # Randomly sample R_c and R_t values
    R_c_values = np.random.normal(rc, rcerr, N)
    R_t_values = np.random.normal(rt, rterr, N)

    # Calculate the half-light radius for each pair of R_c and R_t values
    R_h_values = []
    for R_c, R_t in zip(R_c_values, R_t_values):
        if R_c < R_t and R_c > 0 and R_t > 0:
            rh = np.abs(compute_half_light_radius(R_c, R_t))
            R_h_values.append(rh)

    # Convert to numpy array for easier manipulation
    R_h_values = np.array(R_h_values)

    # The error is the standard deviation of the half-light radii
    sigma_R_h = np.std(R_h_values)

    return sigma_R_h


def main(params_path: str) -> None:
    """
    Main function to perform the computations on the data.

    Args:
        params_path: Path to the params file.
    """
    with open(params_path) as f:
        params = json.load(f)
    rp, rperr = params["plum"]["bestfit"][-1], params["plum"]["bestfit_err"][-1]
    re, reerr = params["expo"]["bestfit"][-1], params["expo"]["bestfit_err"][-1]
    rc, rt = params["king"]["bestfit"][-2], params["king"]["bestfit"][-1]
    rcerr, rterr = params["king"]["bestfit_err"][-2], params["king"]["bestfit_err"][-1]

    params["plum"]["halflight"] = [rp, rperr]
    params["expo"]["halflight"] = [1.68 * re, 1.68 * reerr]
    params["king"]["halflight"] = [np.abs(compute_half_light_radius(rc, rt)), estimate_error_in_half_light_radius(rc, rt, rcerr, rterr)]

    with open(params_path, 'wt') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    params_path = "/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/params.json"
    main(params_path)
