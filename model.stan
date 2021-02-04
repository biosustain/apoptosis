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
  vector[2] prior_sm;
  vector[2] prior_kq;
  vector[2] prior_td;
  vector[2] prior_kd;
  vector[2] prior_sd_ac_sm;
  vector[2] prior_sd_ac_kq;
  vector[2] prior_sd_ac_td;
  vector[2] prior_sd_ac_kd;
  vector[2] prior_R0;
  vector[2] prior_err;
  int<lower=0,upper=1> likelihood;
}
parameters {
  vector<lower=0>[R] R0;
  real<lower=0> err;
  // design effects
  vector<upper=-0.00001>[D] sm;
  vector<lower=0>[D] kq;
  vector<lower=0>[D] td;
  vector<lower=0>[D] kd;
  // clone effects
  vector[C] ac_sm;
  vector[C] ac_kq;
  vector[C] ac_td;
  vector[C] ac_kd;
  // sds of clone effects
  vector<lower=0>[D] sd_ac_sm;
  vector<lower=0>[D] sd_ac_kq;
  vector<lower=0>[D] sd_ac_td;
  vector<lower=0>[D] sd_ac_kd;
}
transformed parameters {
  vector<lower=0>[N] yhat;
  for (n in 1:N){
    int r = replicate[n];
    int c = clone[r];
    int d = design[c];
    yhat[n] = yt(t[n],
                 R0[r],
                 sm[d] + ac_sm[c],
                 kq[d] + ac_kq[c],
                 td[d] + ac_td[c],
                 kd[d] + ac_kd[c]);
  }
}
model {
  // direct priors
  target += lognormal_lpdf(R0 | prior_R0[1], prior_R0[2]);
  target += lognormal_lpdf(err | prior_err[1], prior_err[2]);
  target += normal_lpdf(sm | prior_sm[1], prior_sm[2]);
  target += lognormal_lpdf(kq | prior_kq[1], prior_kq[2]);
  target += lognormal_lpdf(td | prior_td[1], prior_td[2]);
  target += lognormal_lpdf(kd | prior_kd[1], prior_kd[2]);
  target += lognormal_lpdf(sd_ac_sm | prior_sd_ac_sm[1], prior_sd_ac_sm[2]);
  target += lognormal_lpdf(sd_ac_kq | prior_sd_ac_kq[1], prior_sd_ac_kq[2]);
  target += lognormal_lpdf(sd_ac_td | prior_sd_ac_td[1], prior_sd_ac_td[2]);
  target += lognormal_lpdf(sd_ac_kd | prior_sd_ac_kd[1], prior_sd_ac_kd[2]);
  // multilevel priors
  target += normal_lpdf(ac_sm | 0, sd_ac_sm[design]);
  target += normal_lpdf(ac_kq | 0, sd_ac_kq[design]);
  target += normal_lpdf(ac_td | 0, sd_ac_td[design]);
  target += normal_lpdf(ac_kd | 0, sd_ac_kd[design]);
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
