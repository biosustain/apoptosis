import numpy as np
from scipy.stats import norm, lognorm


def get_lognormal_params_from_qs(x1, x2, p1, p2):
    """Find parameters for a lognormal distribution from two quantiles.

    i.e. get mu and sigma such that if X ~ lognormal(mu, sigma), then pr(X <
    x1) = p1 and pr(X < x2) = p2.

    """
    logx1 = np.log(x1)
    logx2 = np.log(x2)
    denom = norm.ppf(p2) - norm.ppf(p1)
    sigma = (logx2 - logx1) / denom
    mu = (logx1 * norm.ppf(p2) - logx2 * norm.ppf(p1)) / denom
    return mu, sigma


def get_normal_params_from_qs(x1, x2, p1, p2):
    """find parameters for a normal distribution from two quantiles.

    i.e. get mu and sigma such that if x ~ normal(mu, sigma), then pr(x <
    x1) = p1 and pr(x < x2) = p2.

    """
    denom = norm.ppf(p2) - norm.ppf(p1)
    sigma = (x2 - x1) / denom
    mu = (x1 * norm.ppf(p2) - x2 * norm.ppf(p1)) / denom
    return mu, sigma


def get_99_pct_params_ln(x1, x2):
    return get_lognormal_params_from_qs(x1, x2, 0.01, 0.99)


def get_99_pct_params_n(x1, x2):
    return get_normal_params_from_qs(x1, x2, 0.01, 0.99)
