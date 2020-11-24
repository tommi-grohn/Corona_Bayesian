functions {
  real[] sir(real t, real[] y, real[] theta, 
             real[] x_r, int[] x_i) {
    
      int N = x_i[1];
      int n_days = x_i[2];
    
      real beta = theta[1];
      real D_e = theta[n_days + 1];
      real D_i = theta[n_days + 2];

      real S = y[1];
      real E = y[2];
      real I = y[3];
      real R = y[4];
      
      real dS_dt;
      real dE_dt;
      real dI_dt;
      real dR_dt;
      
      for (i in 2:n_days) {
        if (i <= t) {
          beta = theta[i];
        }
      }

      dS_dt = -beta * I * S / N;
      dE_dt =  beta * I * S / N - E / D_e;
      dI_dt =  E / D_e - I / D_i;
      dR_dt =  I / D_i;
      
      return {dS_dt, dE_dt, dI_dt, dR_dt};
  }
}

data {
  int<lower=1>  n_days;
  real<lower=0> y0[4];  // 4 stages
  real<lower=0> t0;
  real<lower=0> ts[n_days];
  int<lower=1>  N;
  
  real<lower=0> D_e;
  real<lower=0> D_i;
  real<lower=0> alpha; // Death rate
  
  real<lower=0> traffic1[n_days];
  real<lower=0> traffic2[n_days];
  // real traffic3[n_days];
  // real traffic4[n_days];
  // real traffic5[n_days];
  
  int<lower=0> deaths[n_days];
}

transformed data {
  real x_r[0];
  int  x_i[2] = { N,  n_days};
}

parameters {
  real c[2];
}

transformed parameters{
  real y[n_days, 4];
  real beta[n_days];
  real x[n_days];  // seir-modeled deaths
  real<lower=0> lambda [n_days];  // seir-modeled deaths

  
  for (i in 1:n_days) {
    beta[i] = c[1] * traffic1[i] + c[2] * traffic2[i];
  }
  
  {
    real theta[n_days + 2];
    theta[1:n_days] = beta;
    theta[n_days+1] = D_e;
    theta[n_days+2] = D_i;

    y = integrate_ode_rk45(sir, y0, t0, ts, theta, x_r, x_i);
  }
  
  for (i in 1:n_days) {
    // x[i] = alpha * y[i,4];
    lambda[i] = alpha * y[i,3] / D_i;
  }
  
}

model {
  //priors
  c ~ normal(0, 10);
  
  //sampling distribution
  for (i in 1:n_days) {
      deaths[i] ~ poisson(lambda[i]);
  }
}

generated quantities {
}