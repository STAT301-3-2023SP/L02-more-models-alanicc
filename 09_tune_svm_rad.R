# {svm rad] tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(doParallel)
library(LiblineaR)
library(doMC)
library(kernlab)

# handle common conflicts
tidymodels_prefer()

# set up parallel processing
parallel::detectCores()
registerDoParallel()

# load required objects ----
load("results/tuning_setup.rda")


# Define model ----
svm_rad_mod <- svm_rbf(
  mode = "classification",
  cost = tune(),
  rbf_sigma = tune()) %>% 
  set_engine("kernlab")

# set-up tuning grid ----
svm_rad_params <- extract_parameter_set_dials(svm_rad_mod)

# create grid 
svm_rad_grid <- grid_regular(svm_rad_params, levels = 5)


#update recipe
wildfire_interact <-wildfire_rec %>% 
  step_interact(~all_numeric_predictors()^2)

wildfire_interact %>%
  prep(wildfire_train) %>%
  bake(new_data = NULL)


# workflow ----
svm_rad_workflow <- workflow() %>%
  add_model(svm_rad_mod) %>%
  add_recipe(wildfire_rec)

set.seed(1234)

# Tuning/fitting ----
tic.clearlog()
tic("radial svm")


svm_rad_tune <- tune_grid(
  svm_rad_workflow,
  resamples = wildfire_folds,
  grid = svm_rad_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"),
  metrics = metric_set(roc_auc, f_meas)
)

toc(log = TRUE)

time_log <- tic.log(format = FALSE)

svm_rad_tictoc <- tibble(model = time_log[[1]]$msg,
                         runtime = time_log[[1]]$toc - time_log[[1]]$tic)

save(svm_rad_tune, svm_rad_tictoc,
     file = "results/tuning_svm_rad.rda")

