import arviz as az
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from cmdstanpy import CmdStanModel
from cmdstanpy.utils import jsondump
from itertools import product
from util import get_99_pct_params_ln, get_99_pct_params_n

PRIORS = {
    # design level parameters
    "prior_mu": get_99_pct_params_ln(0.65, 0.73),
    "prior_kq": get_99_pct_params_ln(1, 5),
    "prior_td": get_99_pct_params_ln(0.2, 3),
    "prior_kd": get_99_pct_params_ln(0.3, 4),
    # clone level sds
    "prior_sd_ac_kq": get_99_pct_params_ln(0.03, 0.5),
    "prior_sd_ac_td": get_99_pct_params_ln(0.03, 0.5),
    "prior_sd_ac_kd": get_99_pct_params_ln(0.03, 0.5),
    # initial densities
    "prior_R0": get_99_pct_params_ln(2, 3),
    # measurement error
    "prior_err": get_99_pct_params_ln(0.1, 0.3),
}
OUTPUT_DIR = os.path.join("results", "samples")
PLOT_DIR = "results"
SAMPLE_CONFIG = dict(
    show_progress=True,
    output_dir=OUTPUT_DIR,
    save_warmup=True,
    inits=0,
    chains=4,
    seed=12345
)
CSV_FILE = os.path.join("raw_data", "AllFlaskData_corrected.csv")
MPL_STYLE = "sparse.mplstyle"
STAN_FILE = "model.stan"
LIKELIHOOD = True
TREATMENT = "15ug/mL Puromycin"

    
def plot_qs(infd, params, coord):
    q_table = (
        pd.DataFrame({p: infd.posterior[p].to_series() for p in params})
        .groupby(coord)
        .quantile([0.025, 0.25, 0.5, 0.75, 0.975])
        .stack()
        .reset_index()
        .rename(columns={"level_1": "quantile", "level_2": "parameter", 0: "value"})
    )
    n_param = q_table["parameter"].nunique()
    n_design = q_table[coord].nunique()
    design_order = (
        q_table
        .sort_values("value")
        .set_index(["parameter", "quantile"])
        .loc[(params[0], 0.5), coord]
    )
    f, axes = plt.subplots(1, 3, sharey=True, sharex=True, figsize=[20, 9])
    axes = axes.ravel()
    param_order = dict(zip(params, range(3)))
    for p, df in q_table.groupby("parameter"):
        ax = axes[param_order[p]]
        y = np.linspace(0, 1, n_design)
        dqs = df.set_index([coord, "quantile"])["value"].unstack().loc[design_order]
        ax.set_title(p)
        lines = ax.hlines(y, dqs[0.025], dqs[0.975], color="black")
        ax.set_yticks(y)
        if ax == axes[0]:
            ax.set_yticklabels(list(dqs.index))
    f.suptitle("2.5%-97.5% posterior intervals for design effects")
    plt.tight_layout()
    return f, axes
            
def plot_timecourses(dt, infd):
    y_timecourses = (
        dt.set_index(["design", "clone", "replicate", "day"])["y"].unstack()
    )
    yhat_timecourses = (
        infd.posterior["yhat"].to_series().unstack().reset_index(drop=True).T
        .join(dt[["design", "clone", "replicate", "day"]])
        .set_index(["design", "clone", "replicate", "day"])
        .stack()
        .unstack("day")
    )
    R0s = (
        infd.posterior["R0"].to_series().unstack()
        .reset_index(drop=True).T.stack().rename(0)
    )
    timecourse_groups = (
        yhat_timecourses
        .join(R0s)
        .sort_index(axis=1)
        .groupby(["design", "clone", "replicate"])
    )
    clone_to_row = dt.groupby("clone")[["design"]].first().groupby("design").cumcount().to_dict()
    design_to_col = dict(zip(dt["design"].unique(), range(dt["design"].nunique())))
    f, axes = plt.subplots(
        max(clone_to_row.values())+1, max(design_to_col.values())+1,
        sharex=True, sharey=True, figsize=[20, 10]
    )
    for (design, clone, replicate), df in timecourse_groups:
        row = clone_to_row[clone]
        col = design_to_col[design]
        ax = axes[row, col]
        y = y_timecourses.loc[(design, clone, replicate)]
        samples = df.sample(10)
        for _, sample in samples.iterrows():
            simline = ax.plot(sample, alpha=0.1, color="black", linewidth=0.5)
        yline = ax.plot(y, color="black")
        if row == 0:
            ax.set_title(design)
        if col == 0:
            ax.set_ylabel("Cell density")
        if row == 3:
            ax.set_xlabel("Time (days)")
    f.legend(
        [yline[0], simline[0]], ["observation", "sampled model realisation"],
        frameon=False, loc="upper left", ncol=2
    )
    f.suptitle("Modelled vs observed timecourses")
    return f, axes


def stan_factorize(s_in, first=None):
    values = list(s_in.unique())
    codes = range(1, len(values) + 1)
    if first is not None:
        if first not in values:
            raise ValueError(f"{fist} is not one of the values.")
        values = [first] + [v for v in values if v != first]
    return s_in.map(dict(zip(values, codes)))


def get_design_table_from_raw(raw: pd.DataFrame, treatment: str) -> pd.DataFrame:
    return (
        pd.DataFrame({
            "design": raw["Plasmid"].astype("string"),
            "clone": raw["Clone"].astype("string"),
            "treatment": raw["Treatment"].astype("string"),
            "replicate": raw["Clone"].str.cat(raw["Run"], sep="-").astype("string"),
            "day": raw["Day"].astype(float),
            "y": raw["VCD"].astype(float),
        })
        .loc[lambda df: (
            df["treatment"].eq(treatment)
            & df["day"].gt(0)
            & ~df["design"].eq("None")
        )]
        .assign(
            design_fct=lambda df: stan_factorize(df["design"], first="Empty"),
            clone_fct=lambda df: stan_factorize(df["clone"]),
            replicate_fct=lambda df: stan_factorize(df["replicate"]),
        )
    )


def main():
    plt.style.use(MPL_STYLE)
    model = CmdStanModel(stan_file=STAN_FILE)
    raw = pd.read_csv(CSV_FILE)
    dt = get_design_table_from_raw(raw, treatment=TREATMENT)
    input_data = {**PRIORS, **{
        "N": len(dt),
        "R": dt["replicate"].nunique(),
        "C": dt["clone"].nunique(),
        "D": dt["design"].nunique(),
        "design": dt.groupby("clone_fct")["design_fct"].first().values,
        "clone": dt.groupby("replicate_fct")["clone_fct"].first().values,
        "replicate": dt["replicate_fct"].values,
        "t": dt["day"].values,
        "y": dt["y"].values,
        "likelihood": int(LIKELIHOOD)
    }}
    jsondump("input_data.json", input_data)
    mcmc = model.sample(data=input_data, **SAMPLE_CONFIG)
    mcmc.diagnose()
    infd = az.from_cmdstan(
        mcmc.runset.csv_files,
        log_likelihood="llik",
        coords={
            "design": dt.groupby("design_fct")["design"].first(),
            "design_non_control": dt.groupby("design_fct")["design"].first().loc[2:],
            "clone": dt.groupby("clone_fct")["clone"].first(),
            "replicate": dt.groupby("replicate_fct")["replicate"].first(),
            "dt_ix": dt.index
        },
        dims={
            "R0": ["replicate"],
            "dq": ["design"],
            "dt": ["design"],
            "dd": ["design"],
            "dq_non_control": ["design_non_control"],
            "dt_non_control": ["design_non_control"],
            "dd_non_control": ["design_non_control"],
            "cq": ["clone"],
            "ct": ["clone"],
            "cd": ["clone"],
            "yrep": ["dt_ix"],
            "yhat": ["dt_ix"],
            "llik": ["dt_ix"],
        }
    )
    print(az.summary(infd, var_names=[
        "mu", "err", "qconst", "tconst", "dconst", "sd_cq", "sd_ct", "sd_cd",
    ]))
    print(az.loo(infd, pointwise=True))
    f, axes = plot_qs(
        infd, ["dt_non_control", "dq_non_control", "dd_non_control"],
        "design_non_control"
    )
    f.savefig(os.path.join(PLOT_DIR, "design_param_qs.png"), bbox_inches="tight")
    plt.close("all")
    f, axes = plot_timecourses(dt, infd)
    f.savefig(os.path.join(PLOT_DIR, "timecourses.png"), bbox_inches="tight")
    plt.close("all")
    
        
if __name__ == "__main__":
    main()
