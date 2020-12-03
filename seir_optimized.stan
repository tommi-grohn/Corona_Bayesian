functions {
  real[] seir(real t, real[] y, real[] theta, 
             real[] x_r, int[] x_i) {
    
      int N = x_i[1];
      int n_training = x_i[2];
      int n_tcomponents = x_i[3];
      real D_e = x_r[n_tcomponents *n_training + 1];
      real D_i = x_r[n_tcomponents *n_training + 2];
      vector[n_tcomponents] traffic;

      real S = y[1];
      real E = y[2];
      real I = y[3];
      real R = y[4];
      
      real dS_dt;
      real dE_dt;
      real dI_dt;
      real dR_dt;
      real beta = theta[1];

      int i = 1;
      while (i < t) {
        i = i + 1;
      }

      if (n_tcomponents > 0) {
        beta = beta + to_vector(x_r[(i-1) * n_tcomponents + 1:(i-1) * n_tcomponents + n_tcomponents])' * to_vector(theta[2:]);
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
  int  x_i[3] = { N,  n_training, n_tcomponents};
  real x_r[n_tcomponents*n_training+2];
  for (i in 1:n_training) {
    for (j in 1:n_tcomponents) {
      x_r[j + (i-1) * n_tcomponents] = traffic[i, j];
    }
  }
  x_r[n_tcomponents *n_training + 1] = D_e;
  x_r[n_tcomponents *n_training + 2] = D_i;
}


parameters {
  real traffic_coeff[n_tcomponents+1];
}

transformed parameters{
  real y[n_training, 4];
  vector<lower=0>[n_training] lambda ;  // seir-modeled deaths
  
  y = integrate_ode_rk45(seir, y0, t0, t_training, traffic_coeff, x_r, x_i);
  lambda = alpha * to_vector(y[,3]) / D_i;
}

model {
  //priors
  traffic_coeff ~ normal(0,1); // Reasonable looking, weakly informative?  
  
  //sampling distribution
  deaths ~ poisson(lambda);
}

generated quantities {
  int deaths_hat[n_training] = poisson_rng(lambda);
}
