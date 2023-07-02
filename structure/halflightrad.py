import numpy as np
from scipy.optimize import fsolve
import json

def SigmaKingIntegral(t, a):
    return (0.5 * ((t**2 - 4 * np.sqrt(1 + a**2) * np.sqrt(1 + t**2)) / (1 + a**2) + np.log(1 + t**2)))

def half_light_radius(R_c, R_t):
    a = R_t / R_c
    equation = lambda xh : SigmaKingIntegral(xh, a)- 0.5 * (SigmaKingIntegral(a, a)+SigmaKingIntegral(0, a))
    xh_initial_guess = 20
    xh_solution = fsolve(equation, xh_initial_guess)
    return xh_solution[0] * R_c

def mc_kingerr(rc, rt, rcerr, rterr):
    # Number of simulations
    N = 100

    # Randomly sample R_c and R_t values
    R_c_values = np.random.normal(rc, rcerr, N)
    R_t_values = np.random.normal(rt, rterr, N)

    # Calculate the half-light radius for each pair of R_c and R_t values
    R_h_values = []
    for R_c, R_t in zip(R_c_values, R_t_values):
        if R_c < R_t and R_c > 0 and R_t > 0:
            rh = np.abs(half_light_radius(R_c, R_t))
            R_h_values.append(rh)

    # Convert to numpy array for easier manipulation
    R_h_values = np.array(R_h_values)

    # The error is the standard deviation of the half-light radii
    sigma_R_h = np.std(R_h_values)

    #print(f"The estimated error in the half-light radius is {sigma_R_h}")
    return sigma_R_h

def main():
    params_path = "/Users/akiratokiwa/Git/HSCSextansPMMeasurement/configs/params.json"
    with open(params_path) as f:
        params = json.load(f)
    rp, rperr = params["plum"]["bestfit"][-1], params["plum"]["bestfit_err"][-1]
    re, reerr = params["expo"]["bestfit"][-1], params["expo"]["bestfit_err"][-1]
    rc, rt = params["king"]["bestfit"][-2], params["king"]["bestfit"][-1]
    rcerr, rterr = params["king"]["bestfit_err"][-2], params["king"]["bestfit_err"][-1]

    params["plum"]["halflight"] = [rp, rperr]
    params["expo"]["halflight"] = [1.68* re, 1.68* reerr]
    params["king"]["halflight"] = [np.abs(half_light_radius(rc, rt)), mc_kingerr(rc, rt, rcerr, rterr)]

    with open(params_path, 'wt') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()