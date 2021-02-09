/* Model for a single treatment */

functions {
#include functions.stan
}
data {
  int<lower=1> N;                     // number of observations
  int<lower=1> D;                     // number of designs
  int<lower=1> C;                     // number of clones
  int<lower=1> R;                     // number of replicates (i.e. n total cultures)
  int<lower=1,upper=D> design[C];     // map of clone to design
  int<lower=1,upper=C> clone[R];      // map of replicate to clone
  int<lower=1,upper=R> replicate[N];  // map of observation to replicate
  vector<lower=0>[N] t;
  vector<lower=0>[N] y;
  vector[2] prior_mu;
  vector[2] prior_kq;
  vector[2] prior_td;
  vector[2] prior_kd;
  vector[2] prior_R0;
  vector[2] prior_err;
  int<lower=0,upper=1> likelihood;
}
parameters {
  vector<lower=0>[R] R0;
  real<lower=0> err;
  real<lower=0,upper=prior_kq[1]> mu;
  real qconst;
  real tconst;
  real dconst;
  // sds of clone effects
  real<lower=0> sd_cq;
  real<lower=0> sd_ct;
  real<lower=0> sd_cd;
  // design effects
  vector[D-1] dq_non_control;
  vector[D-1] dt_non_control;
  vector[D-1] dd_non_control;
  // clone effects
  vector<multiplier=sd_cq>[C] cq;
  vector<multiplier=sd_ct>[C] ct;
  vector<multiplier=sd_cd>[C] cd;
}
transformed parameters {
  vector[D] dq = append_row([0]', dq_non_control);
  vector[D] dt = append_row([0]', dt_non_control);
  vector[D] dd = append_row([0]', dd_non_control);
  vector<lower=0>[N] yhat;
  for (n in 1:N){
    int r = replicate[n];
    int c = clone[r];
    int d = design[c];
    yhat[n] = yt(t[n],
                 R0[r],
                 mu,
                 exp(qconst + dq[d] + cq[c]),
                 exp(tconst + dt[d] + ct[c]),
                 exp(dconst + dd[d] + cd[c]));
  }
}
model {
  // direct priors
  R0 ~ lognormal(prior_R0[1], prior_R0[2]);
  err ~ lognormal(prior_err[1], prior_err[2]);
  mu ~ lognormal(prior_mu[1], prior_mu[2]);
  qconst + dq[design] + cq ~ normal(prior_kq[1], prior_kq[2]);
  tconst + dt[design] + ct ~ normal(prior_td[1], prior_td[2]);
  dconst + dd[design] + ct ~ normal(prior_kd[1], prior_kd[2]);
  // priors for multilevel sds
  sd_cq ~ normal(0, 0.1);
  sd_ct ~ normal(0, 0.1);
  sd_cd ~ normal(0, 0.1);
  // multilevel priors
  dq_non_control ~ normal(0, 1);
  dt_non_control ~ normal(0, 1);
  dd_non_control ~ normal(0, 1);
  cq ~ normal(0, sd_cq);
  ct ~ normal(0, sd_ct);
  cd ~ normal(0, sd_cd);
  // likelihood
  if (likelihood){target += lognormal_lpdf(y | log(yhat), err);}
}
generated quantities {
  vector[N] yrep;
  vector[N] llik;
  for (n in 1:N){
    yrep[n] = lognormal_rng(log(yhat[n]), err);
    llik[n] = lognormal_lpdf(y[n] | log(yhat[n]), err);
  }
}
