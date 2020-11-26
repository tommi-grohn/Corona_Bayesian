functions {
  vector tc_gp(row_vector[] t_pred,
                    vector y_data,
                    row_vector[] t_data,
                    real alpha,
                    real rho,
                    real sigma,
                    real delta) {
    int n = num_elements(t_data);
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
  
  real[] sir(real t, real[] y, real[] theta, 
             real[] x_r, int[] x_i) {
               
      real S = y[1];
      real I = y[2];
      real R = y[3];
      real N = x_i[1];
      int n_days = x_i[2];
      row_vector[1] time_pred[1];
      vector[1] tc_pred;
      vector[n_days] tc = to_vector(x_r[:n_days]);
      row_vector[1] linspace[n_days];
      real beta;
      real gamma = theta[2];
      real alpha = x_r[2*n_days+1];
      real rho = x_r[2*n_days+2];
      real sigma = x_r[2*n_days+3];
      real dS_dt;
      real dI_dt;
      real dR_dt;
      time_pred[1][1] = t;
      linspace[,1] = x_r[n_days+1:2*n_days];
      
      tc_pred = tc_gp(time_pred,
                    tc,
                    linspace, 
                    alpha,
                    rho,
                    sigma,
                    1e-9);
      
      beta = theta[1] * tc_pred[1];
      
      dS_dt = -beta * I * S / N;
      dI_dt =  beta * I * S / N - gamma * I;
      dR_dt =  gamma * I;
      
      return {dS_dt, dI_dt, dR_dt};
	  }
}

data {
  int<lower=1> n_days;
  real y0[3];
  real t0;
  real ts[n_days];
  real traffic_component[n_days];
  real linspace[n_days];
  int N;
  int cases[n_days];
  real alpha;
  real sigma;
  real rho;
}

transformed data {
  real delta = 1e-9;
  real x_r[2*n_days+3]; 
  int x_i[2] = { N, n_days };
  x_r[:n_days]= traffic_component;
  x_r[n_days+1:2*n_days]= linspace;
  x_r[2*n_days+1:] = {alpha, rho, sigma};
}

parameters {
  real<lower=delta> gamma;
  real<lower=delta> beta;
  real<lower=delta> phi_inv;
}
transformed parameters{
  real y[n_days, 3];
  real phi = 1. / phi_inv;
  {
    real theta[2];
    theta[1] = beta;
    theta[2] = gamma;

    y = integrate_ode_rk45(sir, y0, t0, ts, theta, x_r, x_i);
  }
}

model {
  //priors
  beta ~ normal(2, 1);
  gamma ~ normal(0.4, 0.5);
  phi_inv ~ exponential(5);
  
  //sampling distribution
  //col(matrix x, int n) - The n-th column of matrix x. Here the number of infected people 
  cases ~ neg_binomial_2(col(to_matrix(y), 2), phi);
}

generated quantities {
  real R0 = beta / gamma;
  real recovery_time = 1 / gamma;
  real pred_cases[n_days];
  pred_cases = neg_binomial_2_rng(col(to_matrix(y), 2), phi);
}
