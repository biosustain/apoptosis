functions {
#include functions.stan
}
data {
  int<lower=1> N;                     // number of training observations
  int<lower=1> N_test;                // number of test observations
  int<lower=1> D;                     // number of designs
  int<lower=1> C;                     // number of clones
  int<lower=1> R;                     // number of replicates (i.e. n total cultures)
  int<lower=1,upper=D> design[C];      // map of clone to design
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
  real dconst;
  real tconst;
  real<lower=0,upper=exp(qconst)> mu;
  // design effects
  vector[D-1] dq_free;
  vector[D-1] dt_free;
  vector[D-1] dd_free;
  // clone effects
  vector[C] cq;
  vector[C] cd;
  vector[C] ct;
}
transformed parameters {
  vector[N] err;
  vector<lower=0>[N] yhat;
  vector[D] dq = append_row(0, dq_free);
  vector[D] dt = append_row(0, dt_free);
  vector[D] dd = append_row(0, dd_free);
  vector[C] log_kq = qconst + dq[design] + cq;
  vector[C] log_td = tconst + dt[design] + cd;
  vector[C] log_kd = dconst + dd[design] + ct;
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
  log_kq ~ normal(prior_kq[1], prior_kq[2]);
  log_td ~ normal(prior_td[1], prior_td[2]);
  log_kd ~ normal(prior_kd[1], prior_kd[2]);
  dq_free ~ normal(0, 0.3);
  dt_free ~ normal(0, 0.3);
  dd_free ~ normal(0, 0.3);
  qconst ~ normal(0, 1);
  tconst ~ normal(0, 1);
  dconst ~ normal(0, 1);
  cq ~ normal(0, 0.1);
  cd ~ normal(0, 0.1);
  ct ~ normal(0, 0.1);
  // likelihood
  if (likelihood){
    rep_vector(2.5, R) ~ lognormal(log(R0), exp(mu_err));
    y ~ lognormal(log(yhat), err);
  }
}
generated quantities {
  vector[R] llik = rep_vector(0, R);
  vector[N_test] yrep;
  vector[D] avg_delay = exp(tconst + dt) + inv(exp(dconst + dd));
  vector[D] tauD = exp(tconst + dt);
  vector[D] k_d = exp(dconst + dd);
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
