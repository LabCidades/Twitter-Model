# API doc can be found here: https://github.com/turicas/covid19-br/blob/master/api.md

library(here)
library(dplyr)
library(lubridate)
library(readr)

data <- read_csv("https://data.brasil.io/dataset/covid19/caso_full.csv.gz")


# Getting only state stuff
state <- data %>%
  filter(is.na(city)) %>%
  select(-city)

# Getting Brazil's national-level data
br <- state %>%
  group_by(date) %>%
  summarise(
    estimated_population_2019 = sum(estimated_population_2019),
    last_available_confirmed_per_100k_inhabitants = sum(last_available_confirmed_per_100k_inhabitants),
    last_available_deaths = sum(last_available_deaths),
    new_confirmed = sum(new_confirmed),
    new_deaths = sum(new_deaths)
  )

br_2020 <- br %>%
  filter(date < "2021-01-01")

# save files
write_csv(state, here("data", "brazil_state.csv"))
write_csv(br, here("data", "brazil_nation.csv"))
write_csv(br_2020, here("data", "brazil_nation_2020.csv"))
