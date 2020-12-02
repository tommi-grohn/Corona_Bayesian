functions {
  real[] seir(real t, real[] y, real[] theta, 
             real[] x_r, int[] x_i) {
    
      int N = x_i[1];
      int n_training = x_i[2];
      
      real D_e = x_r[1];
      real D_p = x_r[2];
      real D_i = x_r[3];
      real r   = x_r[4];
      real r_a = x_r[5];
      real r_p = x_r[6];

      real S = y[1];
      real E = y[2];
      real A = y[3];
      real P = y[4];
      real I = y[5];
      real R = y[6];
      
      real dS_dt;
      real dE_dt;
      real dA_dt;
      real dP_dt;
      real dI_dt;
      real dR_dt;

      real beta = theta[1];
      
      for (i in 2:n_training) {
        if (i <= t) {
          beta = theta[i];
        }
      }

      dS_dt = - (r_a*beta*S*A/N + r_p*beta*S*P/N + beta*S*I/N);
      dE_dt =  r_a*beta*S*A/N + r_p*beta*S*P/N + beta*S*I/N - E/D_e;
      dA_dt =  r*E/D_e - A/D_i;
      dP_dt =  (1-r)*E/D_e - P/D_p;
      dI_dt =  P/D_p - I/D_i;
      dR_dt =  A/D_i + I/D_i;
      
      return {dS_dt, dE_dt, dA_dt, dP_dt, dI_dt, dR_dt};
  }
}

data {
  int<lower=1>  n_training;
  int<lower=0>  n_tcomponents;
  real<lower=0> y0[6];  // 6 stages
  
  real<lower=0> t0;
  real<lower=0> t_training[n_training];
  
  int<lower=1>  N;
  
  real<lower=0> D_e;   // average exposure time 
  real<lower=0> D_p;   // average presymptomatic time 
  real<lower=0> D_i;   // average infected time
  real<lower=0> r;     // proportion of asymptotic cases out of all infected
  real<lower=0> r_a;   // coefficient of asymptomatics
  real<lower=0> r_p;   // coefficient of presymptomatics
  real<lower=0> alpha; // death rate
  
  int<lower=0> deaths[n_training];
  matrix[n_training, n_tcomponents] traffic;
}

transformed data {
  real<lower=0> x_r[6] = {D_e, D_p, D_i, r, r_a, r_p};
  int<lower=0>  x_i[2] = { N, n_training};
}

parameters {
  vector<lower=0>[n_tcomponents+1] traffic_coeff;
}
 
transformed parameters{
  real<lower=0> y[n_training, 6];
  real<lower=0> theta[n_training];
  vector<lower=0>[n_training] lambda;  // seir-modeled deaths

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
  lambda = 0.005 * to_vector(y[,5]) / 20;
}

model {
  //priors
  traffic_coeff ~ uniform(0,0.4); // Reasonable looking, weakly informative?  
  
  //sampling distribution
  deaths ~ poisson(lambda);
}

generated quantities {
  int<lower=0> deaths_hat[n_training] = poisson_rng(lambda);
 }
