# {mlp] tuning ----

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
nn_mod <- mlp(
  mode = "classification", # or regression
  hidden_units = tune(),
  penalty = tune()
) %>%
  set_engine("nnet")


# set-up tuning grid ----
nn_params <- extract_parameter_set_dials(nn_mod)

# create grid 
nn_grid <- grid_regular(nn_params, levels = 5)


#update recipe
wildfire_interact <-wildfire_rec %>% 
  step_interact(~all_numeric_predictors()^2)

wildfire_interact %>%
  prep(wildfire_train) %>%
  bake(new_data = NULL)


# workflow ----
mlp_workflow <- workflow() %>% 
  add_model(nn_mod) %>% 
  add_recipe(wildfire_interact)

set.seed(1234)

# Tuning/fitting ----
tic.clearlog()
tic("mlp")


mlp_tune <- tune_grid(
  mlp_workflow,
  resamples = wildfire_folds,
  grid = nn_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"))

toc(log = TRUE)

time_log <- tic.log(format = FALSE)

mlp_tictoc <- tibble(model = time_log[[1]]$msg,
                     runtime = time_log[[1]]$toc - time_log[[1]]$tic)


save(mlp_tune, mlp_tictoc, mlp_workflow,
     file = "results/tuning_mlp.rda")
