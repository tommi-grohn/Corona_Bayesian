functions {
  real[] seir(real t, real[] y, real[] theta, 
             real[] x_r, int[] x_i) {
    
      int N = x_i[1];
      
      int first_index = x_i[2]; // depends if we are fitting on training or prediction data
      int last_index = x_i[3];  // depends if we are fitting on training or prediction data
      
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

      real beta = theta[first_index]; // initialize beta
      
      for (i in first_index+1:last_index) {
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
  int<lower=1>  n_training;  // number of training data points
  int<lower=0>  n_prediction; // number of predicting data points
  int<lower=0>  n_tcomponents; // number of traffic components
  
  real<lower=0> y0[4];  // 4 stages
  
  real<lower=0> t0;  // starting point of the time interval
  real<lower=0> t[n_training + n_prediction];  // time points, both for training and prediction
  
  int<lower=1>  N;  // starting point
  
  real<lower=0> D_e;  // average exposure time 
  real<lower=0> D_i;  // average infected time
  real<lower=0> alpha;  // death rate
  
  int<lower=0> deaths[n_training + n_prediction];  // deaths, both for training and prediction
  
  matrix<lower=0>[n_training + n_prediction, n_tcomponents] traffic; // traffic components, both for training and prediction
}

transformed data {
  int<lower=1> n_total = n_training + n_prediction;  // number of all data points

  real x_r[2] = {D_e, D_i};
  int  x_i_training[3] = { N, 1,  n_prediction};
  int  x_i_prediction[3] = { N, n_prediction+1, n_total};
}

parameters {
  vector<lower=0>[n_tcomponents+1] traffic_coeff;
}

transformed parameters{
  real<lower=0> y_training[n_training, 4];
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
  lambda_training = alpha * to_vector(y_training[,3]) / D_i;
}

model {
  //priors
  traffic_coeff ~ normal(1,1); // Reasonable looking, weakly informative?  
  
  //sampling distribution
  deaths[1:n_training] ~ poisson(lambda_training);
}

generated quantities {

  int<lower=0> deaths_training_hat[n_training] = poisson_rng(lambda_training);
  
  real<lower=0> y_prediction[n_prediction, 4] = integrate_ode_rk45(seir, y_training[n_training,], t[n_training], t[n_training+1:], 
                                        theta, x_r, x_i_prediction); 
  
  vector<lower=0>[n_prediction] lambda_prediction = alpha * to_vector(y_prediction[,3]) / D_i;
  int<lower=0> deaths_prediction_hat[n_prediction] = poisson_rng(lambda_prediction);  
  
}
