library(cmdstanr)
library(dplyr)
library(here)
library(loo)
library(rstan)
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

# LOO-CV
# Extract pointwise log-likelihood
# using merge_chains=FALSE returns an array, which is easier to
# use with relative_eff()
log_lik_deaths_only <- extract_log_lik(rstan_deaths_only, merge_chains = FALSE)
r_eff_deaths_only <- relative_eff(exp(log_lik_deaths_only), cores = 2)

log_lik_twitter_deaths <- extract_log_lik(rstan_twitter_deaths,
                                          merge_chains = FALSE)
r_eff_twitter_deaths <- relative_eff(exp(log_lik_twitter_deaths), cores = 2)

loo_deaths_only <- loo(log_lik_deaths_only,
                       r_eff = r_eff_deaths_only,
                       cores = 2)
loo_twitter_deaths <- loo(log_lik_twitter_deaths,
                          r_eff = r_eff_twitter_deaths,
                          cores = 2)

# Compare Models
comp <- loo_compare(loo_deaths_only, loo_twitter_deaths)
print("LOO: Deaths Only(1) vs Twitter + Deaths(2)")
print(comp, simplify = FALSE, digits = 3)
