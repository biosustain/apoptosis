/* Model for a single treatment */

functions {
#include functions.stan
}
data {
  int<lower=1> N;                     // number of observations
  int<lower=1> D;                     // number of designs
  int<lower=1> C;                     // number of clones
  int<lower=1> R;                     // number of replicates (i.e. n total cultures)
  int<lower=1> K;                     // number of design parameters
  matrix<lower=0,upper=1>[C, K] x_clone;
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
  real<lower=0> err;
  vector<lower=0>[R] R0;
  real qconst;
  real<lower=0,upper=exp(qconst)> mu;
  real tconst;
  real dconst;
  // design effects
  vector[K] dt;
  vector[K] dd;
  // clone effects
  matrix[C, 3] cv_z;
  vector<lower=0>[3] sd_cv;
  cholesky_factor_corr[3] L_cv;
}
transformed parameters {
  matrix[C, 3] cv = cv_z * diag_pre_multiply(sd_cv, L_cv);
  vector<lower=0>[N] yhat;
  vector[C] log_kq = qconst + cv[,1];
  vector[C] log_td = tconst + x_clone * dt + cv[,2];
  vector[C] log_kd = dconst + x_clone * dd + cv[,3];
  for (n in 1:N){
    int r = replicate[n];
    int c = clone[r];
    yhat[n] = yt(t[n], R0[r], mu, exp(log_kq[c]), exp(log_td[c]), exp(log_kd[c]));
  }
}
model {
  // direct priors
  err ~ lognormal(prior_err[1], prior_err[2]);
  R0 ~ lognormal(log(2.5), err);
  mu ~ lognormal(prior_mu[1], prior_mu[2]);
  qconst ~ normal(prior_kq[1], prior_kq[2]);
  tconst ~ normal(prior_td[1], prior_td[2]);
  dconst ~ normal(prior_kd[1], prior_kd[2]);
  // priors for multilevel sds
  sd_cv ~ lognormal(-2.1, 0.35);  // ~99% of prior mass between 0.05 and 0.25
  L_cv ~ lkj_corr_cholesky(2);
  // multilevel priors
  dt ~ normal(0, 0.3);
  dd ~ normal(0, 0.3);
  to_vector(cv_z) ~ normal(0, 1);
  // likelihood
  if (likelihood){
    log(y) ~ student_t(4, log(yhat), err);
  }
}
generated quantities {
  vector[N] yrep;
  vector[R] llik = rep_vector(0, R);
  matrix[3,3] corr_cv = L_cv * L_cv';
  for (n in 1:N){
    yrep[n] = exp(student_t_rng(4, log(yhat[n]), err));
    llik[replicate[n]] += student_t_lpdf(log(y[n]) | 4, log(yhat[n]), err);
  }
}
