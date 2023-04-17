# {mars] tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(earth)

# handle common conflicts
tidymodels_prefer()

# set up parallel processing
library(doMC)

parallel::detectCores()
registerDoMC(cores = 12)

# load required objects ----
load("results/tuning_setup.rda")


# Define model ----
mars_mod <- mars(
  mode = "classification", # or "regression"
  num_terms = tune(),
  prod_degree = tune()
) %>%
  set_engine("earth")


# set-up tuning grid ----
mars_params <- extract_parameter_set_dials(mars_mod)%>% 
  update(num_terms = num_terms(c(1, 15))) 

# create grid 
mars_grid <- grid_regular(mars_params, levels = 5)

#update recipe
wildfire_interact <- wildfire_rec %>% 
  step_interact(~all_numeric_predictors()^2)

wildfire_interact %>%
  prep(wildfire_train) %>%
  bake(new_data = NULL)


# workflow ----
mars_workflow <- workflow() %>% 
  add_model(mars_mod) %>% 
  add_recipe(wildfire_interact)

set.seed(1234)

# Tuning/fitting ----
tic.clearlog()
tic("mars")


mars_tune <- tune_grid(
  mars_workflow,
  resamples = wildfire_folds,
  grid = mars_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"))

toc(log = TRUE)

time_log <- tic.log(format = FALSE)

mars_tictoc <- tibble(model = time_log[[1]]$msg,
                     runtime = time_log[[1]]$toc - time_log[[1]]$tic)


save(mars_tune, mars_tictoc,
     file = "results/tuning_mars.rda")