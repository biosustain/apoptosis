from loo_compare import compare
from cmdstanpy import CmdStanModel
import arviz as az
from fit_models import (
    get_stan_input,
    PRIORS,
    CSV_FILE,
    TREATMENTS,
    X_COLS,
    STAN_FILES,
    LOO_DIR,
    INFD_DIR
)

SAMPLE_CONFIG = dict(
    show_progress=False,
    save_warmup=False,
    inits=0,
    iter_warmup=300,
    iter_sampling=300,
    chains=2,
    seed=12345
)
K_THRESHOLD = 0.7


class CustomSamplingWrapper(az.SamplingWrapper):
    def __init__(self, msmts, priors, x_cols, **super_kwargs):
        self.msmts = msmts
        self.priors = priors
        self.x_cols = x_cols
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
        original = get_stan_input(self.msmts, self.priors, self.x_cols)
        m_test = self.msmts.loc[lambda df: df["replicate_fct"].eq(idx[0] + 1)]
        m_train = self.msmts.drop(m_train.index)
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
        for model_name, stan_file in STAN_FILES.items():
            for xname, x_cols in X_COLS.items():
                run_name = f"{treatment_label}_{model_name}_{xname}"
                loo_file = os.path.join(LOO_DIR, f"loo_{run_name}.pkl")
                infd_file = os.path.join(INFD_DIR, f"infd_{run_name}.ncdf")
                print("Running reloo analysis for model {run_name}...")
                model = CmdStanModel(stan_file=stan_file)
                msmts = prepare_data(pd.read_csv(CSV_FILE), treatment=treatment)
                loo_orig = pd.from_pickle(loo_file)
                infd_orig = pd.from_pickle(loo_file)
                infd_kwargs = get_infd_kwargs(msmts, x_cols)
                msmts_raw = pd.read_csv(CSV_FILE)
                infds[run_name] = infd
                sw = CustomSamplingWrapper(
                    model=model,
                    idata_orig=infd_orig,
                    loo_orig=loo_orig,
                    sample_kwargs=SAMPLE_CONFIG,
                    idata_kwargs=infd_kwargs,
                    msmts=measurements,
                    priors=PRIORS,
                    x_cols=x_cols
                )
                rl = az.reloo(sw, loo_orig=loo_orig, k_thresh=K_THRESHOLD)
                rl.to_pickle(os.path.join(LOO_DIR, f"reloo_{run_name}.pkl"))
                loos[run_name] = rl
        comparison = compare(loos)
        print("Loo comparison for model {run_name}:")
        print(comparison)
        comparison.to_csv(
            os.path.join(LOO_DIR, f"reloo_comparison_{treatment_label}.csv")
        )


if __name__ == "__main__":
    main()
