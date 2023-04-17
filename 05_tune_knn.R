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
knn_mod <- nearest_neighbor(mode = "classification",
                              neighbors = tune()) %>% 
  set_engine("kknn")

# set-up tuning grid ----
neighbors()

knn_params <- parameters(knn_mod)

# create grid 
knn_grid <- grid_regular(knn_params, levels = 5)


#update recipe
wildfire_interact <-wildfire_rec %>% 
  step_interact(~all_numeric_predictors()^2)

wildfire_interact %>%
  prep(wildfire_train) %>%
  bake(new_data = NULL)


# workflow ----
knn_workflow <- workflow() %>% 
  add_model(knn_mod) %>% 
  add_recipe(wildfire_interact)

# Tuning/fitting ----
tic.clearlog()
tic("knn")


knn_tune <- tune_grid(
  knn_workflow,
  resamples = wildfire_folds,
  grid = knn_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"))

toc(log = TRUE)

time_log <- tic.log(format = FALSE)

knn_tictoc <- tibble(model = time_log[[1]]$msg,
                       runtime = time_log[[1]]$toc - time_log[[1]]$tic)


save(knn_tune, knn_tictoc,
     file = "results/tuning_knn.rda")
