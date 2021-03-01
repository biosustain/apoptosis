functions {
#include functions.stan
}
data {
  int<lower=1> N;                     // number of training observations
  int<lower=1> N_test;                // number of test observations
  int<lower=1> D;                     // number of designs
  int<lower=1> C;                     // number of clones
  int<lower=1> R;                     // number of replicates (i.e. n total cultures)
  int<lower=1> K;                     // number of design parameters
  matrix<lower=0,upper=1>[C, K] x_clone;
  int<lower=1,upper=C> clone[R];      // map of replicate to clone
  int<lower=1,upper=R> replicate[N];  // map of observation to replicate
  vector<lower=0>[N] t;
  vector<lower=0>[N] y;
  int<lower=1,upper=R> replicate_test[N_test];
  vector<lower=0>[N_test] t_test;
  vector<lower=0>[N_test] y_test;
  vector[2] prior_mu;
  vector[2] prior_kq;
  vector[2] prior_td;
  vector[2] prior_kd;
  vector[2] prior_R0;
  vector[2] prior_err;
  int<lower=0,upper=1> likelihood;
}
parameters {
  real mu_err;
  real b_err;
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
  vector[N] err;
  matrix[C, 3] cv = cv_z * diag_pre_multiply(sd_cv, L_cv);
  vector<lower=0>[N] yhat;
  vector[C] log_kq = qconst + cv[,1];
  vector[C] log_td = tconst + x_clone * dt + cv[,2];
  vector[C] log_kd = dconst + x_clone * dd + cv[,3];
  {
    vector[N] x_small;
    for (n in 1:N){
        int r = replicate[n];
        int c = clone[r];
        yhat[n] = yt(t[n], R0[r], mu, exp(log_kq[c]), exp(log_td[c]), exp(log_kd[c]));
        x_small[n] = yhat[n] > 0.3 ? 0 : log(yhat[n] / 0.3);
    }
    err = exp(mu_err + b_err * x_small);
  }
}
model {
  // direct priors
  mu_err ~ normal(prior_err[1], prior_err[2]);
  b_err ~ normal(0, 1);
  R0 ~ lognormal(prior_R0[1], prior_R0[2]);
  mu ~ lognormal(prior_mu[1], prior_mu[2]);
  // qconst ~ normal(prior_kq[1], prior_kq[2]);
  // tconst ~ normal(prior_td[1], prior_td[2]);
  // dconst ~ normal(prior_kd[1], prior_kd[2]);
  log_kq ~ normal(prior_kq[1], prior_kq[2]);
  log_td ~ normal(prior_td[1], prior_td[2]);
  log_kd ~ normal(prior_kd[1], prior_kd[2]);
  dt ~ normal(0, 0.3);
  dd ~ normal(0, 0.3);
  // multilevel priors
  sd_cv ~ lognormal(-1.61, 0.298);  // ~99% of prior mass between 0.1 and 0.4
  L_cv ~ lkj_corr_cholesky(2);
  to_vector(cv_z) ~ normal(0, 1);
  // likelihood
  if (likelihood){
    rep_vector(2.5, R) ~ lognormal(log(R0), exp(mu_err));
    y ~ lognormal(log(yhat), err);
  }
}
generated quantities {
  matrix[3,3] corr_cv = L_cv * L_cv';
  vector[R] llik = rep_vector(0, R);
  vector[N_test] yrep;
  for (n in 1:N_test){
    int r = replicate_test[n];
    int c = clone[r];
    real yhat_test =
      yt(t_test[n], R0[r], mu, exp(log_kq[c]), exp(log_td[c]), exp(log_kd[c]));
    real xs = yhat_test > 0.3 ? 0 : log(yhat_test / 0.3);
    real err_test = exp(mu_err + b_err * xs);
    yrep[n] = lognormal_rng(log(yhat_test), err_test);
    llik[r] += lognormal_lpdf(y_test[n] | log(yhat_test), err_test);
  }
}
