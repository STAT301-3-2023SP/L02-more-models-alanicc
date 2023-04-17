# {svm poly] tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(LiblineaR)
library(doMC)
library(kernlab)

# handle common conflicts
tidymodels_prefer()

# set up parallel processing
library(doMC)

parallel::detectCores()
registerDoMC(cores = 12)

# load required objects ----
load("results/tuning_setup.rda")


# Define model ----
svm_poly_mod <- svm_poly(
  mode = "classification",
  cost = tune(),
  degree = tune(),
  scale_factor = tune()) %>% 
  set_engine("kernlab")

# set-up tuning grid ----
svm_poly_params <- extract_parameter_set_dials(svm_poly_mod)

# create grid 
svm_poly_grid <- grid_regular(svm_poly_params, levels = 5)


#update recipe
wildfire_interact <-wildfire_rec %>% 
  step_interact(~all_numeric_predictors()^2)

wildfire_interact %>%
  prep(wildfire_train) %>%
  bake(new_data = NULL)


# workflow ----
svm_poly_workflow <- workflow() %>%
  add_model(svm_poly_mod) %>%
  add_recipe(wildfire_rec)

set.seed(1234)

# Tuning/fitting ----
tic.clearlog()
tic("polynomial svm")


svm_poly_tune <- tune_grid(
  svm_poly_workflow,
  resamples = wildfire_folds,
  grid = svm_poly_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"),
  metrics = metric_set(roc_auc, f_meas)
)

toc(log = TRUE)

time_log <- tic.log(format = FALSE)

svm_poly_tictoc <- tibble(model = time_log[[1]]$msg,
                          runtime = time_log[[1]]$toc - time_log[[1]]$tic)

save(svm_poly_tune, svm_poly_tictoc,
     file = "results/tuning_svm_poly.rda")
