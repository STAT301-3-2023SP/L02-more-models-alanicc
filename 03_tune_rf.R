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
rf_mod <- rand_forest(mode = "classification",
                        min_n = tune(),
                        mtry = tune()) %>% 
  set_engine("ranger", importance = "impurity")

# set-up tuning grid ----
rf_params <- parameters(rf_mod) %>% 
  # N := maximum number of random predictor columns we want to try 
  # should be less than the number of available columns
  update(mtry = mtry(c(1, 6))) 

# create grid 
rf_grid <- grid_regular(rf_params, levels = 5)


#update recipe
wildfire_interact <-wildfire_rec %>% 
  step_interact(~all_numeric_predictors()^2)

wildfire_interact %>%
  prep(wildfire_train) %>%
  bake(new_data = NULL)

# workflow ----
rf_workflow <- workflow() %>% 
  add_recipe(wildfire_interact) %>% 
  add_model(rf_mod)

# Tuning/fitting ----
tic.clearlog()
tic("random forest")


rf_tune <- tune_grid(
  rf_workflow,
  resamples = wildfire_folds,
  grid = rf_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"))

toc(log = TRUE)

time_log <- tic.log(format = FALSE)

rf_tictoc <- tibble(model = time_log[[1]]$msg,
                    runtime = time_log[[1]]$toc - time_log[[1]]$tic)


save(rf_tune, rf_tictoc,
     file = "results/tuning_rf.rda")
