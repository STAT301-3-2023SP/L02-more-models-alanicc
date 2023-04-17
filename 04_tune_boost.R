# {random forest] tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)


# handle common conflicts
tidymodels_prefer()

# set up parallel processing
library(doMC)

parallel::detectCores()
registerDoMC(cores = 12)

# load required objects ----
load("results/tuning_setup.rda")


# Define model ----
boost_mod <- boost_tree(mode = "classification",
                          mtry = tune(),
                          min_n = tune(),
                          learn_rate = tune()) %>% 
  set_engine("xgboost")

# set-up tuning grid ----
learn_rate()

mtry()

boost_params <- parameters(boost_mod) %>% 
  update(mtry = mtry(range = c(1,15)))

# create grid 
boost_grid <- grid_regular(boost_params, levels = 5)


#update recipe
wildfire_interact <-wildfire_rec %>% 
  step_interact(~all_numeric_predictors()^2)

wildfire_interact %>%
  prep(wildfire_train) %>%
  bake(new_data = NULL)


# workflow ----
boost_workflow <- workflow() %>% 
  add_recipe(wildfire_interact) %>% 
  add_model(boost_mod)

# Tuning/fitting ----
tic.clearlog()
tic("boost")


boost_tune <- tune_grid(
  boost_workflow,
  resamples = wildfire_folds,
  grid = boost_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"))

toc(log = TRUE)

time_log <- tic.log(format = FALSE)

boost_tictoc <- tibble(model = time_log[[1]]$msg,
                    runtime = time_log[[1]]$toc - time_log[[1]]$tic)


save(boost_tune, boost_tictoc,
     file = "results/tuning_boost.rda")
