functions {
#include functions.stan
}
data {
  int<lower=1> N;
  vector[N] t;
  real R0;
  real sm;
  real kq;
  real td;
  real kd;
}
generated quantities {
  vector[N] y_n;
  vector[N] y_a;
  for (n in 1:N){
    y_n[n] = yt_num(t[n], R0, sm, kq, td, kd);
    y_a[n] = yt(t[n], R0, sm, kq, td, kd);
  }
}
