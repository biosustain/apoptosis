import arviz as az
from matplotlib import pyplot as plt
import os
import pandas as pd

from munging import prepare_data
from plotting import plot_qs, plot_timecourses
from run_model import CSV_FILE, NCDF_FILE, TREATMENT

MPL_STYLE = "sparse.mplstyle"
PLOT_DIR = "results"

def main():
    az.rcParams["data.load"] = "eager"  # stop this code from hogging the ncdf file

    plt.style.use(MPL_STYLE)
    measurements = prepare_data(pd.read_csv(CSV_FILE), treatment=TREATMENT)
    infds = []
    names = ["ab", "abc"]
    for name in names:
        print(f"Analysing model {name}...")
        infd = az.from_netcdf(f"infd_{name}.ncdf")
        infds.append(infd)
        loo = az.loo(infd, pointwise=True)
        loo_df = pd.DataFrame({
            "clone": measurements.groupby("replicate")["clone"].first(),
            "design": measurements.groupby("replicate")["design"].first(),
            "pareto_k": loo.pareto_k.to_series(),
            "elpd": loo.loo_i.to_series(),
        })
        print(loo_df.sort_values("pareto_k"))
        # plots
        f, axes = plot_qs(infd, ["dt", "dd"], "design")
        f.savefig(
            os.path.join(PLOT_DIR, f"design_param_qs_{name}.png"),
            bbox_inches="tight"
        )
        f, axes = plot_timecourses(measurements, infd)
        f.savefig(
            os.path.join(PLOT_DIR, f"timecourses_{name}.png"),
            bbox_inches="tight"
        )
        plt.close("all")
    comparison = az.compare(dict(zip(names, infds)), ic="loo")
    print(comparison)


if __name__ == "__main__":
    main()
