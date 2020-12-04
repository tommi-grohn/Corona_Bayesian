functions {
  real[] seapir(real t, real[] y, real[] theta, 
             real[] x_r, int[] x_i) {
    
      int N = x_i[1];
      int n_training = x_i[2];
      int n_tcomponents = x_i[3];

      int data_place = (n_tcomponents + 1)*n_training;
      int component_place;
      
      real tc[n_training] = x_r[n_training* n_tcomponents + 1:(n_tcomponents + 1)*n_training];
      real D_e = x_r[data_place + 1];
      real D_p = x_r[data_place + 2];
      real D_i = x_r[data_place + 3];
      real r   = x_r[data_place + 4];
      real r_a = x_r[data_place + 5];
      real r_p = x_r[data_place + 6];

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

      int i = 1;
      while (tc[i] < t)
        i = i + 1; 

      component_place = (i-1) * n_tcomponents;

      if (n_tcomponents > 0) {
        beta = beta + to_vector(x_r[component_place + 1:component_place + n_tcomponents])' * to_vector(theta[2:]);
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
  int<lower=0>  n_test;
  int<lower=0>  n_tcomponents;
  real<lower=0> y0[6];  // 6 stages
  real<lower=0> t0;
  real<lower=0> t_training[n_training];
  real<lower=0> t_test[n_test];
  int<lower=1>  N;
  
  real<lower=0> D_e;   // average exposure time 
  real<lower=0> D_p;   // average presymptomatic time 
  real<lower=0> D_i;   // average infected time
  real<lower=0> r;     // proportion of asymptotic cases out of all infected
  real<lower=0> r_a;   // coefficient of asymptomatics
  real<lower=0> r_p;   // coefficient of presymptomatics
  real<lower=0> alpha; // death rate
  
  int<lower=0> deaths[n_training];
  int<lower=0> deaths_pred[n_test];
  matrix[n_training, n_tcomponents] traffic;
  matrix[n_test, n_tcomponents] traffic_pred;
}

transformed data {
  int n_sum = n_training + n_test;
  int  x_i[3] = { N, n_training, n_tcomponents};
  real x_r[(n_tcomponents + 1)*n_training+6];
  for (i in 1:n_training) {
    for (j in 1:n_tcomponents) {
      x_r[j + (i-1) * n_tcomponents] = traffic[i, j];
    }
  }
  x_r[n_training * n_tcomponents + 1:(n_tcomponents + 1)*n_training] = t_training;
  x_r[(n_tcomponents + 1)*n_training + 1:(n_tcomponents + 1)*n_training + 6] = {D_e, D_p, D_i, r, r_a, r_p};
}

parameters {
  real traffic_coeff[n_tcomponents+1];
}

transformed parameters{
  real<lower=1e-9> y[n_training, 6];
  vector<lower=1e-9>[n_training] lambda ;  // seir-modeled deaths

  y = integrate_ode_rk45(seapir, y0, t0, t_training, traffic_coeff, x_r, x_i);
  lambda = 0.008 * to_vector(y[,5]) / 20;  
}

model {
  //priors
  traffic_coeff ~ normal(0,10); // Reasonable looking, weakly informative?  
  
  //sampling distribution
  deaths ~ poisson(lambda);
}

generated quantities {
  real x_r_test[(n_tcomponents + 1)*n_sum+6];
  int x_i_test[3];
  real<lower=1e-9> y_hat[n_sum, 6];
  int deaths_hat[n_sum];
  vector[n_sum] lambda_hat;
  real log_lik;

  x_i_test = { N, n_sum, n_tcomponents };
  for (i in 1:n_sum) {
    for (j in 1:n_tcomponents) {
      if (i < n_training + 1) {
        x_r_test[j + (i-1) * n_tcomponents] = traffic[i, j];
      }
      else {
        x_r_test[j + (i-1) * n_tcomponents] = traffic_pred[i-n_training, j];
      }
    }
  }
  x_r_test[n_sum * n_tcomponents + 1:(n_tcomponents + 1)*n_sum] = append_array(t_training, t_test);
  x_r_test[(n_tcomponents + 1)*n_sum + 1:(n_tcomponents + 1)*n_sum + 6] = {D_e, D_p, D_i, r, r_a, r_p};

  y_hat = integrate_ode_rk45(seapir, y0, t0, append_array(t_training, t_test), traffic_coeff, x_r_test, x_i_test);
  lambda_hat = 0.008 * to_vector(y_hat[,5]) / 20;
  deaths_hat = poisson_rng(lambda_hat);
  log_lik = poisson_lpmf(deaths_pred | lambda_hat[n_training + 1:]);
}