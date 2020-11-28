functions {
  real[] seir(real t, real[] y, real[] theta, 
             real[] x_r, int[] x_i) {
    
      int N = x_i[1];
      int n_training = x_i[2];
    
      real beta = theta[1];
      real D_e = theta[n_training + 1];
      real D_i = theta[n_training + 2];

      real S = y[1];
      real E = y[2];
      real I = y[3];
      real R = y[4];
      
      real dS_dt;
      real dE_dt;
      real dI_dt;
      real dR_dt;
      
      for (i in 2:n_training) {
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
  int<lower=1>  n_total;
  int<lower=1>  n_training;
  int<lower=1>  n_prediction;
  real<lower=0> y0[4];  // 4 stages
  
  real<lower=0> t0;
  real<lower=0> t_training[n_training];
  real<lower=0> t_prediction[n_prediction];
  
  int<lower=1>  N;
  
  real<lower=0> D_e;
  real<lower=0> D_i;
  real<lower=0> alpha; // Death rate
  
  real<lower=0> traffic[n_total];

  int<lower=0> deaths[n_total];
}

transformed data {
  real x_r[0];
  int  x_i[2] = { N,  n_training};
}

parameters {
  real<lower=0> c[4];
}

transformed parameters{
  real y[n_training, 4];
  real beta[n_training];
  real<lower=0> lambda [n_training];  // seir-modeled deaths

  
  for (i in 1:n_training) {
    beta[i] = c[1] * traffic[i] + c[2]; 
  }
  
  {
    real theta[n_training + 2];
    theta[1:n_training] = beta;
    theta[n_training+1] = D_e;
    theta[n_training+2] = D_i;

    y = integrate_ode_rk45(seir, y0, t0, t_training, theta, x_r, x_i);
  }
  
  for (i in 1:n_training) {
    lambda[i] = alpha * y[i,3] / D_i;
  }
  
}

model {
  //priors
  c ~ normal(1,1); // Reasonable looking, weakly informative?  
  
  //sampling distribution
  for (i in 1:n_training) {
      deaths[i] ~ poisson(lambda[i]);
  }
}

generated quantities {
  // Here should be added the predictions!
  
}