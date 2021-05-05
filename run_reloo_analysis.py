import json
import os

import arviz as az
import pandas as pd
from cmdstanpy import CmdStanModel

from fit_models import (CSV_FILE, INFD_DIR, LOO_DIR, MODEL_SETS, OUTPUT_DIR,
                        PRIORS, STAN_FILES, TREATMENT_TO_MODEL_SET, TREATMENTS,
                        get_infd_kwargs, get_stan_input)
from loo_compare import compare
from munging import prepare_data

SAMPLE_CONFIG = dict(
    show_progress=False,
    save_warmup=False,
    inits=0,
    iter_warmup=200,
    iter_sampling=200,
    chains=1,
    seed=12345,
)
K_THRESHOLD = 0.7


class CustomSamplingWrapper(az.SamplingWrapper):
    def __init__(self, msmts, priors, design_col, **super_kwargs):
        self.msmts = msmts
        self.priors = priors
        self.design_col = design_col
        super(CustomSamplingWrapper, self).__init__(**super_kwargs)

    def sample(self, data):
        """Call CmdStanModel.sample."""
        return self.model.sample(data=data, **self.sample_kwargs)

    def get_inference_data(self, mcmc):
        """Call arviz.from_cmdstanpy."""
        return az.from_cmdstanpy(mcmc, **self.idata_kwargs)

    def log_likelihood__i(self, excluded_obs, idata__i):
        """Get the out-of-sample log likelihoods from idata__i."""
        ll = idata__i.log_likelihood["llik"]
        return ll.where(ll != 0, drop=True)

    def sel_observations(self, idx):
        """Construct a stan input where replicate idx is out-of-sample."""
        original = get_stan_input(self.msmts, self.priors, self.design_col)
        m_test = self.msmts.loc[lambda df: df["replicate_fct"].eq(idx[0] + 1)]
        m_train = self.msmts.drop(m_test.index)
        d_test = original.copy()
        d_test["t"] = m_train["day"].values
        d_test["t_test"] = m_test["day"].values
        d_test["y"] = m_train["y"].values
        d_test["y_test"] = m_test["y"].values
        d_test["replicate"] = m_train["replicate_fct"].values
        d_test["replicate_test"] = m_test["replicate_fct"].values
        d_test["N"] = len(m_train)
        d_test["N_test"] = len(m_test)
        return d_test, {}


def main():
    for treatment_label, treatment in TREATMENTS.items():
        loos = {}
        for model_name, xname in MODEL_SETS[TREATMENT_TO_MODEL_SET[treatment_label]]:
            stan_file = STAN_FILES[model_name]
            design_col = "design_" + xname
            run_name = f"{treatment_label}_{model_name}_{xname}"
            loo_file = os.path.join(LOO_DIR, f"loo_{run_name}.pkl")
            infd_file = os.path.join(INFD_DIR, f"infd_{run_name}.ncdf")
            json_file = os.path.join(OUTPUT_DIR, f"input_data_{run_name}.json")
            print(f"Running reloo analysis for model {run_name}...")
            model = CmdStanModel(stan_file=stan_file)
            msmts = prepare_data(pd.read_csv(CSV_FILE), treatment=treatment)
            loo_orig = pd.read_pickle(loo_file)
            infd_orig = az.from_netcdf(infd_file)
            with open(json_file, "r") as f:
                stan_input = json.load(f)
            infd_kwargs = get_infd_kwargs(msmts, design_col, stan_input)
            sw = CustomSamplingWrapper(
                model=model,
                idata_orig=infd_orig,
                sample_kwargs=SAMPLE_CONFIG,
                idata_kwargs=infd_kwargs,
                msmts=msmts,
                priors=PRIORS,
                design_col=design_col,
            )
            rl = az.reloo(sw, loo_orig=loo_orig, k_thresh=K_THRESHOLD)
            rl.to_pickle(os.path.join(LOO_DIR, f"reloo_{run_name}.pkl"))
            loos[run_name] = rl
        comparison = compare(loos)
        print(f"Loo comparison for model {treatment_label}:")
        print(comparison)
        comparison.to_csv(
            os.path.join(LOO_DIR, f"reloo_comparison_{treatment_label}.csv")
        )


if __name__ == "__main__":
    main()
