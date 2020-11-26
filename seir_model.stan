functions {
  vector tc_gp(row_vector[] t_pred,
                    vector y_data,
                    row_vector[] t_data,
                    real alpha,
                    real rho,
                    real sigma,
                    real delta) {
    int n = rows(y_data);
    int T = size(t_pred);
    real pred_y[T];
    vector[T] pred_mu;
    {
        matrix[T, T] cov_f2;
        matrix[n, n] L_K;
        vector[n] K_div_y1;
        matrix[n, T] k_x1_t;
        matrix[n, T] v_pred;
        matrix[n, n] K;
        K = cov_exp_quad(t_data, alpha, rho);
        for (i in 1:n)
            K[i, i] = K[i, i] + square(sigma);
        L_K = cholesky_decompose(K);
        K_div_y1 = mdivide_left_tri_low(L_K, y_data);
        K_div_y1 = mdivide_right_tri_low(K_div_y1', L_K)';
        k_x1_t = cov_exp_quad(t_data, t_pred, alpha, rho);
        pred_mu = (k_x1_t' * K_div_y1);
        v_pred = mdivide_left_tri_low(L_K, k_x1_t);
        cov_f2 = cov_exp_quad(t_pred, alpha, rho) - v_pred' * v_pred;
        for(i in 1:T)
            cov_f2[i,i] = cov_f2[i,i] + delta;
    }
    return pred_mu;
}

  real[] seir(real t, real[] y, real[] theta, 
             real[] x_r, int[] x_i) {
    
      int N = x_i[1];
      int n_days = x_i[2];

      real beta;
      vector[1] tc_t;
      real D_e = x_r[1];
      real D_i = x_r[2];
      real alpha = x_r[3];
      real sigma = x_r[4];
      real rho = x_r[5];

      vector[n_days] traffic = to_vector(x_r[6:n_days+5]); 
      row_vector[1] ts[n_days]; 

      real S = y[1];
      real E = y[2];
      real I = y[3];
      real R = y[4];
      
      real dS_dt;
      real dE_dt;
      real dI_dt;
      real dR_dt;
      row_vector[1] time_pred[1];
      time_pred[1][1] = t;
      ts[,1] = x_r[n_days+6:];

      tc_t = tc_gp(time_pred,
                    traffic,
                    ts, 
                    alpha,
                    rho,
                    sigma,
                    1e-9);

      beta = sum(to_vector(theta) * tc_t[1]);

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
  
  real<lower=0> traffic[n_days];

  int<lower=0> deaths[n_days];
  real alpha_g;
  real sigma;
  real rho;
}

transformed data {
  int  x_i[2] = { N,  n_days};

  real x_r[5 + 2*n_days];
  x_r[:5]= { D_e, D_i, alpha_g, sigma, rho};
  x_r[6:n_days+5] = traffic;
  x_r[n_days+6:] = ts;
}

parameters {
  real<lower=1e-9> theta[1];
}

transformed parameters{
  real y[n_days, 4];
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
