# {elastic net] tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)

# handle common conflicts
tidymodels_prefer()

# set up parallel processing
parallel::detectCores()
cl <- makePSOCKcluster(12)
registerDoParallel(cl)

stopCluster(cl)

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
wildfire_interact <-wildfire_rec %>% 
  step_interact(~all_numeric_predictors()^2)

wildfire_interact %>%
  prep(wildfire_train) %>%
  bake(new_data = NULL) %>% 
  view()

stopCluster(cl)

# # define tuning grid
# en_tune <- wildfire_wf %>%
#   tune_grid(resamples = wildfire_folds, 
#             grid = en_grid, 
#             metrics = metric_set(mae, rmse))

# workflow ----
wildfire_wf <- workflow() %>% 
  add_recipe(wildfire_rec) %>% 
  add_model(en_mod)

# Tuning/fitting ----
tic.clearlog()
tic("elastic net")


en_tune <- tune_grid(
  wildfire_wwf,
  resamples = wildfire_folds,
  grid = en_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"))

toc(log = TRUE)

time_log <- tic.log(format = FALSE)

en_tictoc <- tibble(model = time_log[[1]]$msg,
                    runtime = time_log[[1]]$toc - time_log[[1]]$tic)

stopCluster(cl)

save(en_tune, en_tictoc,
     file = "results/tuning_en.rda")
