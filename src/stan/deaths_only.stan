data {
  int no_days;
  int population;
  //observed new deaths per day
  array[no_days] int new_deaths;
  int likelihood;
  real beta_regularization;
}
transformed data {
  int no_weeks = no_days %/% 7 + min(1, no_days % 7);
  array[no_weeks] int new_weekly_deaths;
  for (week in 1 : no_weeks) {
    int start = 1 + 7 * (week - 1);
    int end = min(start + 6, no_days);
    new_weekly_deaths[week] = sum(new_deaths[start : end]);
  }
}
parameters {
  /*
    The elements of a simplex sum to one.
    The first no_days entries represent daily new infections as a fraction
    of the total population while the last entry represents the proportion that
    remains susceptible at the end.
  */
  simplex[no_days + 1] unit_dS;
  real<lower=0> dL;
  real<lower=0> dI;
  real<lower=0> dT;
  real<lower=0, upper=1> omega;
  real<lower=0> reciprocal_phi_deaths;
  array[7] real<lower=0, upper=1> reporting_probability;
}
transformed parameters {
  vector[no_days] daily_infections = population * unit_dS[ : no_days];
  vector[no_days] daily_deaths;
  vector[no_weeks] weekly_deaths;
  vector[no_days] beta;
  vector[no_days] effective_reproduction_number;

  // States to be recovered
  vector[no_days] state_S;
  vector[no_days] state_E;
  vector[no_days] state_I;
  vector[no_days] state_T;
  vector[no_days] state_D;

  if (likelihood) {
    vector[4] state = [0, 0, 0, 0]';
    matrix[4, 4] transition_matrix = matrix_exp([//[E     ,I           ,T     ,D]
                                                 [-1 / dL, 0, 0, 0],
                                                 //E
                                                 [+1 / dL, -1 / dI, 0, 0],
                                                 //I
                                                 [0, +1 / dI * omega,
                                                  -1 / dT, 0],
                                                 //T
                                                 [0, 0, +1 / dT, 0] //D
                                                 ]);
    real S = population;
    real last_D;
    int weekday;
    for (i in 1 : no_days) {
      weekday = 1 + (i - 1) % 7;
      last_D = state[4];
      S -= daily_infections[i];
      state[1] += daily_infections[i];
      state = transition_matrix * state;
      daily_deaths[i] = state[4] - last_D;
      beta[i] = daily_infections[i] * population / (S * state[2]); // S * I
      effective_reproduction_number[i] = daily_infections[i] / state[2] * dI; // I

      // Populate States
      state_S[i] = S;
      state_E[i] = state[1];
      state_I[i] = state[2];
      state_T[i] = state[3];
      state_D[i] = state[4];
    }
    for (week in 1 : no_weeks) {
      int start = 1 + 7 * (week - 1);
      int end = min(start + 6, no_days);
      weekly_deaths[week] = sum(daily_deaths[start : end]);
    }
  }
}
model {
  //One possible regularization
  if (beta_regularization) {
    unit_dS[2 : no_days] ~ lognormal(log(unit_dS[ : no_days - 1]),
                                     beta_regularization);
  }
  //This imposes a very wide prior on the proportion of still susceptible people!
  unit_dS[no_days + 1] ~ uniform(0, 1);
  dL ~ normal(4.0, 0.2);
  dI ~ normal(3.06, 0.21);
  dT ~ normal(16.0, 0.71);
  omega ~ beta(100, 9803);
  reciprocal_phi_deaths ~ exponential(5);
  if (likelihood) {
    new_weekly_deaths ~ neg_binomial_2(weekly_deaths,
                                       1 / reciprocal_phi_deaths);
  }
}
generated quantities {
  array[no_weeks] int pred_weekly_deaths;
  array[no_days] int pred_daily_deaths;
  vector[no_weeks] log_lik;

  pred_weekly_deaths = neg_binomial_2_rng(weekly_deaths,
                                          1 / reciprocal_phi_deaths);
  pred_daily_deaths = neg_binomial_2_rng(daily_deaths,
                                         1 / reciprocal_phi_deaths);
  for (i in 1 : no_weeks) {
    log_lik[i] = neg_binomial_2_lpmf(new_weekly_deaths[i] | weekly_deaths[i], 1
                                                                    / reciprocal_phi_deaths);
  }
}
