# Load package(s)
library(tidymodels)
library(tidyverse)
library(skimr)

# handle common conflicts
tidymodels_prefer()

# Seed
set.seed(3013)

## load data
wildfires_dat <- read_csv("data/wildfires.csv") %>%
  janitor::clean_names() %>%
  mutate(
    winddir = factor(winddir, levels = c("N", "NE", "E", "SE", "S", "SW", "W", "NW")),
    traffic = factor(traffic, levels = c("lo", "med", "hi")),
    wlf = factor(wlf, levels = c(1, 0), labels = c("yes", "no"))
  ) %>%
  select(-burned)

# check response variable
# class imbalance - not really
# if class imbalance we can consider downsampling; upsampling

#skim for missingness - none
# if present we need to impute; step_impute_mean... _median, _mode, _knn, _linear
skim_without_charts(wildfires_dat)
skim(wildfires_dat)

# splitting data
set.seed(3013)
wildfire_split <- initial_split(wildfires_dat,
                          prop = 0.8,
                          strata = wlf)

# train and test
wildfire_train <- training(wildfire_split)
wildfire_test <- testing(wildfire_split)

# v-fold cross validation
wildfire_folds <- vfold_cv(wildfire_train, v = 5, repeats = 3)

# recipes
wildfire_rec <- recipe(wlf ~ ., data = wildfire_train) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())

wildfire_rec %>%
  prep(wildfire_train) %>%
  bake(new_data = NULL) %>% 
  view()

save(wildfire_rec, wildfire_folds, wildfire_test, wildfire_train,
     file = "results/tuning_setup.rda")
