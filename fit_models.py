import os

import arviz as az
import pandas as pd
from cmdstanpy import CmdStanModel
from cmdstanpy.utils import get_logger, jsondump

from munging import prepare_data
from util import get_99_pct_params_ln

PRIORS = {
    "prior_mu": get_99_pct_params_ln(0.65, 0.73),
    "prior_kq": get_99_pct_params_ln(1, 5),
    "prior_td": get_99_pct_params_ln(0.4, 7.5),
    "prior_kd": get_99_pct_params_ln(0.05, 2.5),
    "prior_R0": get_99_pct_params_ln(2, 3),
    "prior_err": get_99_pct_params_ln(0.05, 0.13),
    "prior_sd_cv": get_99_pct_params_ln(0.02, 0.2),
}
CSV_FILE = os.path.join("raw_data", "AllFlaskData_compiled.csv")
LIKELIHOOD = 1
OUTPUT_DIR = "results"
LOO_DIR = os.path.join(OUTPUT_DIR, "loo")
INFD_DIR = os.path.join(OUTPUT_DIR, "infd")
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples")
LIKELIHOOD = True

SAMPLE_CONFIG = dict(
    show_progress=False,
    save_warmup=False,
    inits=0,
    iter_warmup=300,
    iter_sampling=300,
    chains=4,
    seed=12345,
    adapt_delta=0.99,
    output_dir=SAMPLES_DIR,
)

TREATMENTS = {
    "tunicamycin": "7.5uM Tunicamycin",
    "brefeldinA": "7.5um Brefeldin A",
    "puromycin": "15ug/mL Puromycin",
    "sodium_butyrate": "20mM Sodium Butyrate",
}
MODEL_SETS = {
    "main": [
        ("m1", "ab_no_interaction"),
        ("m1", "ab"),
        ("m1", "abc"),
        ("m2", "null"),
        ("m2", "ab_no_interaction"),
        ("m2", "ab"),
        ("m2", "abc"),
    ],
    "supplementary": [
        ("m1", "ab"),
        ("m2", "ab"),
        ("m2", "null"),
    ],
}
EFFECT_PRIOR_SDS = {
    "dt": {
        "baseline": 0.3,
        "is_A": 0.3,
        "is_B": 0.3,
        "is_AB": 0.1,
        "is_C": 0.3,
        "is_AC": 0.1,
        "is_BC": 0.1,
        "is_ABC": 0.05,
    },
    "dd": {
        "baseline": 0.3,
        "is_A": 0.3,
        "is_B": 0.3,
        "is_AB": 0.1,
        "is_C": 0.3,
        "is_AC": 0.1,
        "is_BC": 0.1,
        "is_ABC": 0.05,
    },
    "dq": {
        "baseline": 0.3,
        "is_A": 0.3,
        "is_B": 0.3,
        "is_AB": 0.1,
        "is_C": 0.3,
        "is_AC": 0.1,
        "is_BC": 0.1,
        "is_ABC": 0.05,
    },
}
TREATMENT_TO_MODEL_SET = {
    "puromycin": "main",
    "sodium_butyrate": "main",
    "tunicamycin": "supplementary",
    "brefeldinA": "supplementary",
}
X_COLS = {
    "null": ["baseline"],
    "ab_no_interaction": ["baseline", "is_A", "is_B"],
    "ab": ["baseline", "is_A", "is_B", "is_AB"],
    "abc": ["baseline", "is_A", "is_B", "is_AB", "is_C", "is_AC", "is_BC", "is_ABC"],
}
STAN_FILES = {
    "m1": "model_kq_design_effects.stan",
    "m2": "model_no_kq_design_effects.stan",
}


def get_effect_prior_sds(x_clone_cols):
    return


def get_stan_input(msmts, priors, x_clone_cols):
    x_clone = msmts.groupby("clone_fct")[x_clone_cols].first().astype(int)
    return {
        **priors,
        **{
            "prior_sd_dd": [EFFECT_PRIOR_SDS["dd"][c] for c in x_clone_cols],
            "prior_sd_dt": [EFFECT_PRIOR_SDS["dt"][c] for c in x_clone_cols],
            "prior_sd_dq": [EFFECT_PRIOR_SDS["dq"][c] for c in x_clone_cols],
            "N": len(msmts),
            "N_test": len(msmts),
            "R": msmts["replicate"].nunique(),
            "C": msmts["clone"].nunique(),
            "D": msmts["design"].nunique(),
            "K": x_clone.shape[1],
            "design": msmts.groupby("clone_fct")["design_fct"].first().values,
            "clone": msmts.groupby("replicate_fct")["clone_fct"].first().values,
            "x_clone": x_clone.values,
            "replicate": msmts["replicate_fct"].values,
            "t": msmts["day"].values,
            "y": msmts["y"].values,
            "replicate_test": msmts["replicate_fct"].values,
            "t_test": msmts["day"].values,
            "y_test": msmts["y"].values,
            "likelihood": int(LIKELIHOOD),
        },
    }


def get_infd_kwargs(msmts, x_cols, stan_input):
    coords = {
        "clone": msmts.groupby("clone_fct")["clone"].first(),
        "replicate": msmts.groupby("replicate_fct")["replicate"].first(),
        "cv_effects": ["q", "tau", "d"],
    }
    dims = {
        "R0": ["replicate"],
        "cv": ["clone", "cv_effects"],
        "sd_cv": ["cv_effects"],
        "llik": ["replicate"],
    }
    if len(x_cols) > 1:
        coords["design"] = x_cols
        dims["msmts"] = ["design"]
        dims["dd"] = ["design"]
        dims["dt"] = ["design"]
        dims["dq"] = ["design"]
        dims["avg_delay"] = ["design"]
    return dict(
        log_likelihood="llik",
        posterior_predictive="yrep",
        coords=coords,
        dims=dims,
        observed_data={"y": stan_input["y"]},
        constant_data={"x_clone": stan_input["x_clone"]},
        save_warmup=SAMPLE_CONFIG["save_warmup"],
    )


def main():
    logger = get_logger()
    logger.setLevel(40)  # only log messages with at-least-error severity
    for treatment_label, treatment in TREATMENTS.items():
        infds = {}
        for model_name, xname in MODEL_SETS[TREATMENT_TO_MODEL_SET[treatment_label]]:
            stan_file = STAN_FILES[model_name]
            x_cols = X_COLS[xname]
            run_name = f"{treatment_label}_{model_name}_{xname}"
            loo_file = os.path.join(LOO_DIR, f"loo_{run_name}.pkl")
            infd_file = os.path.join(INFD_DIR, f"infd_{run_name}.ncdf")
            json_file = os.path.join(OUTPUT_DIR, f"input_data_{run_name}.json")
            print(f"Fitting model {run_name}...")
            model = CmdStanModel(stan_file=stan_file, logger=logger)
            msmts = prepare_data(pd.read_csv(CSV_FILE), treatment=treatment)
            stan_input = get_stan_input(msmts, PRIORS, x_cols)
            jsondump(json_file, stan_input)
            mcmc = model.sample(data=stan_input, **SAMPLE_CONFIG)
            print(mcmc.diagnose().replace("\n\n", "\n"))
            infd_kwargs = get_infd_kwargs(msmts, x_cols, stan_input)
            infd = az.from_cmdstanpy(mcmc, **infd_kwargs)
            print(az.summary(infd, var_names=["avg_delay", "dt", "dd", "sd_cv"]))
            infds[run_name] = infd
            loo = az.loo(infd, pointwise=True)
            print(f"Writing inference data to {infd_file}")
            infd.to_netcdf(infd_file)
            print(f"Writing psis-loo results to {loo_file}\n")
            loo.to_pickle(loo_file)
        comparison = az.compare(infds)
        print(f"Loo comparison for treatment {treatment}:")
        print(comparison)
        comparison.to_csv(
            os.path.join(LOO_DIR, f"loo_comparison_{treatment_label}.csv")
        )


if __name__ == "__main__":
    main()
