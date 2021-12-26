library(cmdstanr)
library(dplyr)
library(here)
library(readr)

# based on Funko_Unko's contribution: https://discourse.mc-stan.org/t/codatmo-liverpool-uninove-models-slow-ode-implementation-and-trapezoidal-solver/22500/45

# SEITD -------------------------------------------------------------------

model <- cmdstan_model(here("src", "stan", "deaths_only.stan"))

# Real data
br <- read_csv(here("data", "brazil_nation_2020.csv"))

no_days <- br %>% nrow
population <- br %>% pull(estimated_population_2019) %>% max
new_deaths <- br %>% pull(new_deaths)

stan_data <- list(
  no_days = no_days,
  population = population,
  new_deaths = new_deaths,
  likelihood = 1,
  beta_regularization = 0.10,
  model_periodicity = 1
)

fit <- model$sample(data = stan_data,
                    seed = 1729,
                    parallel_chains = 4,
                    output_dir = here("results", "deaths_only"))
