# {elastic net] tuning ----

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
en_mod <-logistic_reg(mode = "classification",
                        penalty = tune(),
                        mixture = tune()) %>% 
  set_engine("glmnet")

# set-up tuning grid ----
# params
en_params <- extract_parameter_set_dials(en_mod)

# create grid 
en_grid <- grid_regular(en_params, levels = 5)

#update recipe
wildfire_interact <- wildfire_rec %>%
  step_interact(~all_numeric_predictors()^2)

wildfire_interact %>%
  prep(wildfire_train) %>%
  bake(new_data = NULL)


# workflow ----
en_workflow <- workflow() %>% 
  add_recipe(wildfire_interact) %>% 
  add_model(en_mod)

# Tuning/fitting ----
tic.clearlog()
tic("elastic net")


en_tune <- tune_grid(
  en_workflow,
  resamples = wildfire_folds,
  grid = en_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"))

toc(log = TRUE)

time_log <- tic.log(format = FALSE)

en_tictoc <- tibble(model = time_log[[1]]$msg,
                    runtime = time_log[[1]]$toc - time_log[[1]]$tic)


save(en_tune, en_tictoc,
     file = "results/tuning_en.rda")

