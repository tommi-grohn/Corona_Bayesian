functions {
  real[] sir(real t, real[] y, real[] theta, 
             real[] x_r, int[] x_i) {
    
      real beta = theta[1];
      real D_e = theta[14 + 1];
      real D_i = theta[14 + 2];

      real S = y[1];
      real E = y[2];
      real I = y[3];
      real R = y[4];
      
      real N = x_i[1];
      
      real dS_dt;
      real dE_dt;
      real dI_dt;
      real dR_dt;
      
      for (i in 2:14) {
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
  int<lower=1> n_days;
  real y0[4];  // 4 stages
  real t0;
  real ts[n_days];
  int N;
  
  real D_e;
  real D_i;
  real alpha; // Death rate
  
  real traffic1[n_days];
  int<lower=0> deaths[n_days];
}

transformed data {
  real x_r[0];
  int x_i[1] = { N };
}

parameters {
  real c1;
}

transformed parameters{
  real y[n_days, 4];
  real beta[n_days];
  real x[n_days];  // seir-modeled deaths
  real<lower=0> lambda [n_days];  // seir-modeled deaths

  
  for (i in 1:n_days) {
    beta[i] = c1 * traffic1[i];
  }
  
  {
    real theta[n_days + 2];
    theta[1:n_days] = beta;
    theta[n_days+1] = D_e;
    theta[n_days+2] = D_i;

    y = integrate_ode_rk45(sir, y0, t0, ts, theta, x_r, x_i);
  }
  
  for (i in 1:n_days) {
    x[i] = alpha * y[i,4];
    lambda[i] = alpha * y[i,3] / D_i;
  }
  
}

model {
  //priors
  c1 ~ normal(0, 10);
  
  //sampling distribution
  //col(matrix x, int n) - The n-th column of matrix x. Here the number of infected people 
  for (i in 1:n_days) {
      deaths[i] ~ poisson(lambda[i]);
  }
  

  // deaths ~ neg_binomial_2(col(to_matrix(y), 2), phi);
}
    
generated quantities {
}