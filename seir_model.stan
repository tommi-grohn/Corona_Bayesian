functions {
  real[] seir(real t, real[] y, real[] theta, 
             real[] x_r, int[] x_i) {
    
      int N = x_i[1];
      
      int first_index = x_i[2]; // depends if we are fitting on training or prediction data
      int last_index = x_i[3];  // depends if we are fitting on training or prediction data
      
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

      real beta = theta[first_index]; // initialize beta
      
      for (i in first_index+1:last_index) {
        if (i <= t) {
          beta = theta[i];
        }
      }

      dS_dt = - (r_a*beta*A/N + r_p*beta*P/N + beta*I/ N);
      dE_dt =  r_a*beta*A/N + r_p*beta*P/N + beta*I/ N - E/D_e;
      dA_dt =  r*E/D_e - A/D_i;
      dP_dt =  (1-r)*E/D_e - P/D_p;
      dI_dt =  P/D_p - I/D_i;
      dR_dt =  A/D_i + I/D_i;
      
      return {dS_dt, dE_dt, dA_dt, dP_dt, dI_dt, dR_dt};
  }
}

data {
  int<lower=1>  n_training;  // number of training data points
  int<lower=0>  n_prediction; // number of predicting data points
  int<lower=0>  n_tcomponents; // number of traffic components
  
  real<lower=0> y0[6];  // 6 stages
  
  real<lower=0> t0;  // starting point of the time interval
  real<lower=0> t[n_training + n_prediction];  // time points, both for training and prediction
  
  int<lower=1>  N;  // starting point
  
  real<lower=0> D_e;  // average exposure time 
  real<lower=0> D_p;  // average presymptomatic time 
  real<lower=0> D_i;  // average infected time
  real<lower=0> r;  // average infected time
  real<lower=0> r_a;  // average infected time
  real<lower=0> r_p;  // average infected time
  real<lower=0> alpha;  // death rate
  
  int<lower=0> deaths[n_training + n_prediction];  // deaths, both for training and prediction
  
  matrix<lower=0>[n_training + n_prediction, n_tcomponents] traffic; // traffic components, both for training and prediction
}

transformed data {
  int<lower=1> n_total = n_training + n_prediction;  // number of all data points

  real x_r[6] = {D_e, D_p, D_i, r, r_a, r_p};
  int  x_i_training[3] = { N, 1,  n_prediction};
  int  x_i_prediction[3] = { N, n_prediction+1, n_total};
}

parameters {
  vector<lower=0>[n_tcomponents+1] traffic_coeff;
}

transformed parameters{
  real<lower=0> y_training[n_training, 6];
  real<lower=0> theta[n_training + n_prediction];  // betas, both for training and prediction
  vector<lower=0>[n_training] lambda_training ;  // seir-modeled deaths

  if (n_tcomponents > 0) {
    vector[n_training  + n_prediction] beta = rep_vector(traffic_coeff[1], n_training + n_prediction);
    beta = beta + traffic * traffic_coeff[2:];
    for (i in 1:n_training+n_prediction) {
      theta[i] = beta[i];
    }
  }
  else {
    theta = rep_array(traffic_coeff[1], n_training + n_prediction);
  }
  
  y_training = integrate_ode_rk45(seir, y0, t0, t[1:n_training], theta, x_r, x_i_training);
  lambda_training = alpha * to_vector(y_training[,5]) / D_i;
}

model {
  //priors
  traffic_coeff ~ normal(1,1); // Reasonable looking, weakly informative?  
  
  //sampling distribution
  deaths[1:n_training] ~ poisson(lambda_training);
}

generated quantities {

  int<lower=0> deaths_training_hat[n_training] = poisson_rng(lambda_training);
  
  real<lower=0> y_prediction[n_prediction, 6] = integrate_ode_rk45(seir, y_training[n_training,], t[n_training], t[n_training+1:], 
                                        theta, x_r, x_i_prediction); 
  vector<lower=0>[n_prediction] lambda_prediction = alpha * to_vector(y_prediction[,5]) / D_i;
  int<lower=0> deaths_prediction_hat[n_prediction] = poisson_rng(lambda_prediction);  
  
}
