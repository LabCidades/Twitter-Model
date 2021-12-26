library(cmdstanr)
library(dplyr)
library(here)
library(readr)

# based on Funko_Unko's contribution: https://discourse.mc-stan.org/t/codatmo-liverpool-uninove-models-slow-ode-implementation-and-trapezoidal-solver/22500/45

# SEITD -------------------------------------------------------------------

model <- cmdstan_model(here("src", "stan", "twitter_deaths.stan"))

# Real data
br <- read_csv(here("data", "brazil_nation_2020.csv"))

no_days <- br %>% nrow
population <- br %>% pull(estimated_population_2019) %>% max
new_deaths <- br %>% pull(new_deaths)

# Twitter Predictions
tweets <- read_csv(here("data", "daily_twitter_pred_2020.csv")) %>%
  filter(date >= "2020-02-25")

new_tweets <- tweets %>% pull(freq)

stan_data <- list(
  no_days = no_days,
  population = population,
  new_deaths = new_deaths,
  new_tweets = new_tweets,
  likelihood = 1,
  beta_regularization = 0.10
)

fit <- model$sample(data = stan_data,
                    seed = 1729,
                    parallel_chains = 4,
                    output_dir = here("results", "twitter_deaths"))
