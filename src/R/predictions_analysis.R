library(cmdstanr)
library(rstan)
library(dplyr)
library(readr)
library(tibble)
library(here)

deaths_only <- list.files(here("results", "deaths_only"),
                               full.names = TRUE) %>%
        as_cmdstan_fit()
rstan_deaths_only <- list.files(here("results", "deaths_only"),
                                     full.names = TRUE) %>%
        read_stan_csv()
twitter_deaths <- list.files(here("results", "twitter_deaths"),
                                  full.names = TRUE) %>%
        as_cmdstan_fit()
rstan_twitter_deaths <- list.files(here("results", "twitter_deaths"),
                                        full.names = TRUE) %>%
        read_stan_csv()

pred_deaths_only <- deaths_only$summary("pred_weekly_deaths")
pred_twitter_deaths <- twitter_deaths$summary("pred_weekly_deaths")

# Real data
br <- read_csv(here("data", "brazil_nation_2020.csv")) %>%
  filter(date <= "2020-12-20")

real_deaths <- br %>%
  group_by(week = cut(date, "week")) %>%
  summarise(deaths = sum(new_deaths)) %>%
  pull(deaths) %>%
  ceiling %>%
  as.integer %>%
  enframe(name = "week", value = "real_deaths")

# MAE Real vs Predicted Deaths Only
# Last week only
pred_deaths_only %>%
  bind_cols(real_deaths) %>%
  tail(1) %>%
  mutate(
    MAE_median = abs(median - real_deaths),
    MAE_mean = abs(mean - real_deaths)) %>%
  summarise(
    MAE_median = mean(MAE_median),
    MAE_mean = mean(MAE_mean))

# MAE Real vs Predicted Deaths Twitter
# Last week only
pred_twitter_deaths %>%
  bind_cols(real_deaths) %>%
  tail(1) %>%
  mutate(
    MAE_median = abs(median - real_deaths),
    MAE_mean = abs(mean - real_deaths)) %>%
  summarise(
    MAE_median = mean(MAE_median),
    MAE_mean = mean(MAE_mean))
