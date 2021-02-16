import arviz as az
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from cmdstanpy import CmdStanModel
from cmdstanpy.utils import jsondump
from itertools import product

from munging import prepare_data
from util import get_99_pct_params_ln, get_99_pct_params_n

PRIORS = {
    # design level parameters
    "prior_mu": get_99_pct_params_ln(0.65, 0.73),
    "prior_kq": get_99_pct_params_ln(1, 5),
    "prior_td": get_99_pct_params_ln(0.2, 5),
    "prior_kd": get_99_pct_params_ln(0.3, 4),
    # initial densities
    "prior_R0": get_99_pct_params_ln(2, 3),
    "prior_err": get_99_pct_params_ln(0.05, 0.13)
}
OUTPUT_DIR = os.path.join("results", "samples")
SAMPLE_CONFIG = dict(
    show_progress=True,
    output_dir=OUTPUT_DIR,
    save_warmup=False,
    inits=0,
    adapt_delta=0.9,
    chains=4,
    seed=12345
)
X_CLONE_COLS_AB = ["is_A", "is_B", "is_AB"]
X_CLONE_COLS_ABC = X_CLONE_COLS_AB + ["is_C", "is_AC", "is_BC", "is_ABC"]
SUMMARY_VARS = ["mu", "qconst", "tconst", "dconst", "sd_cv", "corr_cv", "dt", "dd"]
CSV_FILE = os.path.join("raw_data", "AllFlaskData_compiled.csv")
NCDF_FILE = os.path.join("results", "infd.ncdf")
STAN_FILE = "model.stan"
LIKELIHOOD = True
TREATMENT = "15ug/mL Puromycin"
# MEASUREMENT_ERROR = 0.1


def get_stan_input(msmts, priors, x_clone_cols):
    x_clone = msmts.groupby("clone_fct")[x_clone_cols].first().astype(int)
    return {**priors, **{
        "N": len(msmts),
        "R": msmts["replicate"].nunique(),
        "C": msmts["clone"].nunique(),
        "D": msmts["design"].nunique(),
        "K": x_clone.shape[1],
        "design": msmts.groupby("clone_fct")["design_fct"].first().values,
        "clone": msmts.groupby("replicate_fct")["clone_fct"].first().values,
        "replicate": msmts["replicate_fct"].values,
        "x_clone": x_clone.values,
        "t": msmts["day"].values,
        "y": msmts["y"].values,
        "likelihood": int(LIKELIHOOD),
        # "err": MEASUREMENT_ERROR
    }}


def load_infd(mcmc, msmts, x_clone_cols):
    return az.from_cmdstanpy(
        mcmc,
        log_likelihood="llik",
        coords={
            "design": x_clone_cols,
            "clone": msmts.groupby("clone_fct")["clone"].first(),
            "replicate": msmts.groupby("replicate_fct")["replicate"].first(),
            "msmts_ix": msmts.index,
            "cv_effects": ["q", "tau", "d"]
        },
        dims={
            "R0": ["replicate"],
            "msmts": ["design"],
            "dd": ["design"],
            "dt": ["design"],
            "cv": ["clone", "cv_effects"],
            "sd_cv": ["cv_effects"],
            "yrep": ["msmts_ix"],
            "yhat": ["msmts_ix"],
            "llik": ["replicate"],
        },
        save_warmup=SAMPLE_CONFIG["save_warmup"]
    )


def main():
    model = CmdStanModel(stan_file=STAN_FILE)
    measurements = prepare_data(pd.read_csv(CSV_FILE), treatment=TREATMENT)
    infds = []
    names = ["ab", "abc"]
    for x_clone_cols, name in zip([X_CLONE_COLS_AB, X_CLONE_COLS_ABC], names):
        stan_input = get_stan_input(measurements, PRIORS, x_clone_cols)
        jsondump("input_data.json", stan_input)
        mcmc = model.sample(data=stan_input, **SAMPLE_CONFIG)
        mcmc.diagnose()
        infd = load_infd(mcmc, measurements, x_clone_cols)
        infd.to_netcdf(f"infd_{name}.ncdf")
        infds.append(infd)
        print(az.summary(infd, var_names=SUMMARY_VARS))
    print(az.compare(dict(zip(names, infds)), ic="loo"))
    
        
if __name__ == "__main__":
    main()
