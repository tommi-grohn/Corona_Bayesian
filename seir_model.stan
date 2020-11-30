functions {
  real[] seir(real t, real[] y, real[] theta, 
             real[] x_r, int[] x_i) {
    
      int N = x_i[1];
      int n_training = x_i[2];
      real D_e = x_r[1];
      real D_i = x_r[2];

      real S = y[1];
      real E = y[2];
      real I = y[3];
      real R = y[4];
      
      real dS_dt;
      real dE_dt;
      real dI_dt;
      real dR_dt;

      real beta = theta[1];
      
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
  int<lower=1>  n_training;
  int<lower=0>  n_prediction;
  int<lower=0>  n_tcomponents;
  real<lower=0> y0[4];  // 4 stages
  
  real<lower=0> t0;
  real<lower=0> t_training[n_training];
  real<lower=0> t_prediction[n_prediction];
  
  int<lower=1>  N;
  
  real<lower=0> D_e;
  real<lower=0> D_i;
  real<lower=0> alpha; // Death rate
  int<lower=0> deaths[n_training + n_prediction];
  matrix[n_training, n_tcomponents] traffic;
}

transformed data {
  real x_r[2] = {D_e, D_i};
  int  x_i[2] = { N,  n_training};
}

parameters {
  vector[n_tcomponents+1] traffic_coeff;
}

transformed parameters{
  real y[n_training, 4];
  real theta[n_training];
  vector<lower=0>[n_training] lambda ;  // seir-modeled deaths

  if (n_tcomponents > 0) {
    vector[n_training] beta = rep_vector(traffic_coeff[1], n_training);
    beta = beta + traffic * traffic_coeff[2:];
    for (i in 1:n_training) {
      theta[i] = beta[i];
    }
  }
  else {
    theta = rep_array(traffic_coeff[1], n_training);
  }
  
  y = integrate_ode_rk45(seir, y0, t0, t_training, theta, x_r, x_i);
  lambda = alpha * to_vector(y[,3]) / D_i;
}

model {
  //priors
  traffic_coeff ~ normal(1,1); // Reasonable looking, weakly informative?  
  
  //sampling distribution
  deaths ~ poisson(lambda);
}

generated quantities {
  int deaths_hat[n_training] = poisson_rng(lambda);
}
