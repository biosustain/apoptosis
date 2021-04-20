import arviz as az
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd

from munging import prepare_data
from fit_models import TREATMENTS, STAN_FILES, X_COLS, INFD_DIR, CSV_FILE, LOO_DIR

MPL_STYLE = "sparse.mplstyle"
PLOT_DIR = os.path.join("results", "plots")


def plot_design_qs(infd):
    params = [p for p in ["dt", "dd", "dq"] if p in infd.posterior]
    q_table = (
        pd.DataFrame({p: infd.posterior[p].to_series() for p in params})
        .groupby("design")
        .quantile([0.025, 0.25, 0.5, 0.75, 0.975])
        .stack()
        .reset_index()
        .rename(columns={"level_1": "quantile", "level_2": "parameter", 0: "value"})
    )
    n_param = q_table["parameter"].nunique()
    n_design = q_table["design"].nunique()
    design_order = (
        q_table
        .sort_values("value")
        .set_index(["parameter", "quantile"])
        .loc[(params[0], 0.5), "design"]
    )
    f, axes = plt.subplots(1, len(params), sharey=True, sharex=True, figsize=[12, 8])
    axes = axes.ravel()
    param_order = dict(zip(params, range(3)))
    for p, df in q_table.groupby("parameter"):
        ax = axes[param_order[p]]
        y = np.linspace(0, 1, n_design)
        dqs = df.set_index(["design", "quantile"])["value"].unstack().loc[design_order]
        if isinstance(dqs, pd.Series): dqs = pd.DataFrame(dqs).transpose()
        ax.set_title(p)
        lines = ax.hlines(y, dqs[0.025], dqs[0.975], color="black")
        ax.set_yticks(y)
        if ax == axes[0]:
            ax.set_yticklabels(list(dqs.index))
    for ax in axes:
        ax.axvline(0, color="red")
    f.suptitle(f"2.5%-97.5% posterior intervals for design effects")
    plt.tight_layout()
    return f, axes
            

def plot_timecourses(msmts, infd, run_name):
    y_timecourses = (
        msmts.set_index(["design", "clone", "replicate", "day"])["y"].unstack()
    )
    yhat_timecourses = (
        infd.posterior["yhat"].to_series().unstack().reset_index(drop=True).T
        .set_index(msmts.index)
        .join(msmts[["design", "clone", "replicate", "day"]])
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
    clone_to_row = msmts.groupby("clone")[["design"]].first().groupby("design").cumcount().to_dict()
    design_to_col = dict(zip(msmts["design"].unique(), range(msmts["design"].nunique())))
    f, axes = plt.subplots(
        max(clone_to_row.values())+1, max(design_to_col.values())+1,
        sharex=True, sharey=True, figsize=[20, 10]
    )
    for (design, clone, replicate), df in timecourse_groups:
        row = clone_to_row[clone]
        col = design_to_col[design]
        ax = axes[row, col]
        y = y_timecourses.loc[(design, clone, replicate)]
        low = df.quantile(0.005)
        high = df.quantile(0.995)
        fill = ax.fill_between(
            df.columns, low, high, alpha=0.2, zorder=0, color="grey"
        )
        yline = ax.plot(y, color="black", label="Clone " + clone)
        if row == 0:
            ax.set_title(design)
        if col == 0:
            ax.set_ylabel("Cell density")
        if row == 3:
            ax.set_xlabel("Time (days)")
        ax.annotate("Clone: " + clone, [4, 4], fontsize=7)
        plt.semilogy()
    f.legend(
        [yline[0], fill], ["observation", "99% Posterior predictive interval"],
        frameon=False, loc="upper left", ncol=2
    )
    f.suptitle(f"Modelled vs observed timecourses: {run_name}")
    return f, axes


def main():
    plt.style.use(MPL_STYLE)
    for treatment_label, treatment in TREATMENTS.items():
        for model_name, stan_file in STAN_FILES.items():
            for xname, x_cols in X_COLS.items():
                run_name = f"{treatment_label}_{model_name}_{xname}"
                print(f"Drawing plots for model {run_name}")
                msmts = prepare_data(pd.read_csv(CSV_FILE), treatment=treatment)
                comparison = (
                    pd.read_csv(os.path.join(LOO_DIR, f"reloo_comparison_{treatment_label}.csv"))
                    .rename(columns={"Unnamed: 0": "index"}).set_index("index")
                )
                infd_file = os.path.join(INFD_DIR, f"infd_{run_name}.ncdf")
                infd = az.from_netcdf(infd_file)
                ## Effects
                f, axes = plot_design_qs(infd)
                f.savefig(
                    os.path.join(PLOT_DIR, f"design_param_qs_{run_name}.png"),
                    bbox_inches="tight"
                )
                ## Measurement and simulated timecourse profiles
                f, axes = plot_timecourses(msmts, infd, run_name)
                f.savefig(
                    os.path.join(PLOT_DIR, f"timecourses_{run_name}.svg"),
                    bbox_inches="tight"
                )
                ## LOO (reloo) scores
                az.plot_compare(comparison, insample_dev=False, plot_ic_diff=False)
                plt.xlabel("LOO Score")
                plt.title(f"{treatment_label}")
                plt.savefig(
                    os.path.join(PLOT_DIR, f"model_RELOO_comparison_{treatment_label}.svg"),
                    bbox_inches="tight"
                )
                
                ## KDE and trace of sampled values
                params = [p for p in ["dt", "dd", "dq"] if p in infd.posterior]
                az.plot_trace(infd, var_names=params, legend=True)
                plt.savefig(
                    os.path.join(PLOT_DIR, f"sampled_params_{run_name}.svg"),
                    bbox_inches="tight"
                )
                plt.show()
                plt.close("all")


if __name__ == "__main__":
    main()
