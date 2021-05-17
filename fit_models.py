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
    "puromycin": "15ug/mL Puromycin",
    "sodium_butyrate": "20mM Sodium Butyrate",
    "tunicamycin": "7.5uM Tunicamycin",
    "brefeldinA": "7.5um Brefeldin A",
}
MODEL_SETS = {
    "main": [
        ("m1", "ab"),
        ("m1", "abc"),
        ("m2", "ab"),
        ("m2", "abc"),
        ("null", "null"),
    ],
    "supplementary": [
        ("m2", "ab"),
        ("m2", "abc"),
    ],
}
TREATMENT_TO_MODEL_SET = {
    "puromycin": "main",
    "sodium_butyrate": "main",
    "tunicamycin": "supplementary",
    "brefeldinA": "supplementary",
}
STAN_FILES = {
    "m2": "m2.stan",
    "m1": "m1.stan",
    "null": "null.stan",
}


def get_stan_input(msmts, priors, design_col):
    out = {
        **priors,
        **{
            "N": int(len(msmts)),
            "N_test": int(len(msmts)),
            "R": int(msmts["replicate"].nunique()),
            "C": int(msmts["clone"].nunique()),
            "clone": msmts.groupby("replicate_fct")["clone_fct"].first().values,
            "replicate": msmts["replicate_fct"].values,
            "t": msmts["day"].values,
            "y": msmts["y"].values,
            "replicate_test": msmts["replicate_fct"].values,
            "t_test": msmts["day"].values,
            "y_test": msmts["y"].values,
            "likelihood": int(LIKELIHOOD),
        },
    }
    if "null" not in design_col:
        out["design"] = msmts.groupby("clone_fct")[design_col + "_fct"].first().values
        out["D"] = int(msmts[design_col + "_fct"].max())
    return out


def get_infd_kwargs(msmts, design_col, stan_input):
    coords = {
        "clone": msmts.groupby("clone_fct")["clone"].first(),
        "replicate": msmts.groupby("replicate_fct")["replicate"].first(),
        "cv_effects": ["q", "tau", "d"],
    }
    dims = {
        "R0": ["replicate"],
        "cq": ["clone"],
        "cd": ["clone"],
        "ct": ["clone"],
        "sd_cv": ["cv_effects"],
        "llik": ["replicate"],
    }
    if "null" not in design_col:
        coords["design"] = msmts.groupby(design_col + "_fct")[design_col].first()
        dims["avg_delay"] = ["design"]
        dims["dq"] = ["design"]
        dims["dd"] = ["design"]
        dims["dt"] = ["design"]
    return dict(
        log_likelihood="llik",
        posterior_predictive="yrep",
        coords=coords,
        dims=dims,
        observed_data={"y": stan_input["y"]},
        save_warmup=SAMPLE_CONFIG["save_warmup"],
    )


def main():
    logger = get_logger()
    logger.setLevel(40)  # only log messages with at-least-error severity
    for treatment_label, treatment in TREATMENTS.items():
        infds = {}
        for model_name, xname in MODEL_SETS[TREATMENT_TO_MODEL_SET[treatment_label]]:
            stan_file = STAN_FILES[model_name]
            design_col = "design_" + xname
            run_name = f"{treatment_label}_{model_name}_{xname}"
            loo_file = os.path.join(LOO_DIR, f"loo_{run_name}.pkl")
            infd_file = os.path.join(INFD_DIR, f"infd_{run_name}.nc")
            json_file = os.path.join(OUTPUT_DIR, f"input_data_{run_name}.json")
            print(f"Fitting model {run_name}...")
            model = CmdStanModel(stan_file=stan_file, logger=logger)
            msmts = prepare_data(pd.read_csv(CSV_FILE), treatment=treatment)
            stan_input = get_stan_input(msmts, PRIORS, design_col)
            jsondump(json_file, stan_input)
            mcmc = model.sample(data=stan_input, **SAMPLE_CONFIG)
            print(mcmc.diagnose().replace("\n\n", "\n"))
            infd_kwargs = get_infd_kwargs(msmts, design_col, stan_input)
            infd = az.from_cmdstanpy(mcmc, **infd_kwargs)
            print(
                az.summary(
                    infd,
                    var_names=[
                        "~cq",
                        "~cd",
                        "~ct",
                        "~err",
                        "~yhat",
                        "~dq_free",
                        "~dd_free",
                        "~dt_free",
                        "~R0",
                        "~log_kd",
                        "~log_td",
                        "~log_kq",
                    ],
                )
            )
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
