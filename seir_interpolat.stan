functions {
  vector tc_interpolate(real t, vector ts, matrix tc, int n_tc) {
    int i = 1;
    vector[n_tc] res;
    while(t > ts[i]) {
      i = i + 1;
    }
    if (i==1) {
      res = tc[1,]';
    }
    else {
      real x1 = ts[i-1];
      real x2 = ts[i];
      vector[n_tc] y1 = tc[i-1,]';
      vector[n_tc] y2 = tc[i,]';
      vector[n_tc] k = (y1-y2)/(x1-x2);
      vector[n_tc] b = y1 - k*x1;
      res = k * t + b;
    }
    return res;
  }

  real[] seir(real t, real[] y, real[] theta, 
             real[] x_r, int[] x_i) {
    
      int N = x_i[1];
      int n_days = x_i[2];
      int n_tc = x_i[3];
      int components = size(theta);

      real beta;
      vector[n_tc] tc_t;
      matrix[n_days, n_tc] traffic;
      real D_e = x_r[1];
      real D_i = x_r[2];

      matrix[n_days, n_tc] tc; 
      vector[n_days] ts = to_vector(x_r[n_days*n_tc+1:]); 

      real S = y[1];
      real E = y[2];
      real I = y[3];
      real R = y[4];
      
      real dS_dt;
      real dE_dt;
      real dI_dt;
      real dR_dt;

      for (i in 1:n_days) {
        for (j in 1:n_tc) {
	  traffic[i, j] = x_r[j + (i-1) * n_tc];
	}
      }

      tc_t = tc_interpolate(t, ts, tc, n_tc);

      beta = dot_product(to_vector(theta), tc_t);

      dS_dt = -beta * I * S / N;
      dE_dt =  beta * I * S / N - E / D_e;
      dI_dt =  E / D_e - I / D_i;
      dR_dt =  I / D_i;
      
      return {dS_dt, dE_dt, dI_dt, dR_dt};
  }
}

data {
  int<lower=1>  n_days;
  int<lower=1>  N;
  int<lower=1> n_tc;

  real<lower=0> y0[4];  // 4 stages
  real<lower=0> t0;
  real<lower=0> ts[n_days];
  
  real<lower=0> D_e;
  real<lower=0> D_i;
  real<lower=0> alpha; // Death rate 
  matrix[n_days, n_tc] traffic;
  int<lower=0> deaths[n_days];
}

transformed data {
  int  x_i[3] = { N,  n_days, n_tc};
  real x_r[(n_tc + 1)*n_days];
  for (i in 1:n_days) {
    for (j in 1:n_tc) {
      x_r[j + (i-1) * n_tc] = traffic[i, j];
    }
  }
  x_r[n_days * n_tc + 1:] = ts;
}

parameters {
  real theta[n_tc];
}

transformed parameters{
  real<lower=0> y[n_days, 4];
  real<lower=0> lambda [n_days];  // seir-modeled deaths
  
  
  {
    y = integrate_ode_rk45(seir, y0, t0, ts, theta, x_r, x_i);
  }
  
  for (i in 1:n_days) {
    lambda[i] = alpha * y[i,3] / D_i;
  }
  
}

model {
  //priors
  theta ~ normal(0, 1);
  
  //sampling distribution
  for (i in 1:n_days) {
      deaths[i] ~ poisson(lambda[i]);
  }
}

generated quantities {
}
