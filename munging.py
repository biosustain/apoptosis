import pandas as pd


def stan_factorize(s_in, first=None):
    values = list(s_in.unique())
    codes = range(1, len(values) + 1)
    if first is not None:
        if first not in values:
            raise ValueError(f"{fist} is not one of the values.")
        values = [first] + [v for v in values if v != first]
    return s_in.map(dict(zip(values, codes)))


def prepare_data(raw: pd.DataFrame, treatment: str) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "design": raw["Plasmid"].astype("string"),
                "clone": raw["Clone"].astype("string"),
                "treatment": raw["Treatment"].astype("string"),
                "replicate": raw["Clone"].str.cat(raw["Run"], sep="-").astype("string"),
                "day": raw["Day"].astype(float),
                "y": raw["VCD"].astype(float),
            }
        )
        .loc[
            lambda df: (
                df["treatment"].eq(treatment)
                & df["day"].gt(0)
                & ~df["design"].eq("None")
            )
        ]
        .assign(
            baseline=1,
            design_fct=lambda df: stan_factorize(df["design"], first="Empty"),
            clone_fct=lambda df: stan_factorize(df["clone"]),
            replicate_fct=lambda df: stan_factorize(df["replicate"]),
            is_A=lambda df: df["design"].str.contains("Bak"),
            is_B=lambda df: df["design"].str.contains("Bax"),
            is_C=lambda df: df["design"].str.contains("Bok"),
            is_AB=lambda df: df["is_A"] & df["is_B"],
            is_AC=lambda df: df["is_A"] & df["is_C"],
            is_BC=lambda df: df["is_B"] & df["is_C"],
            is_ABC=lambda df: df["is_A"] & df["is_B"] & df["is_C"],
        )
    )
