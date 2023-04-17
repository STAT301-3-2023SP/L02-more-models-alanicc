# get final results
library(doMC)
library(tidyverse)
library(tidymodels)

tidymodels_prefer()

parallel::detectCores()
registerDoMC(cores = 12)

results_files <- list.files("results/", "*.rda", full.names = TRUE)

for(i in results_files){
  load(i)
}

####################################
# baseline/null model
null_mod <- null_model(mode = "classification") %>% 
  set_engine("parsnip")

null_wkflw <- workflow() %>% 
  add_model(null_mod) %>% 
  add_recipe(wildfire_rec)

null_fit <- null_wkflw %>% 
  fit_resamples(resamples = wildfire_folds,
                control = control_resamples(save_pred = TRUE))

null_results <- null_fit %>% 
  collect_metrics()

####################################
# organize results to find best overall

# individual model results 
# this is something I recommend putting in the appendix
autoplot(en_tune, metric = "roc_auc")

en_tune %>% 
  show_best(metric = "roc_auc")

####################################
# put all tuned grids together
model_set <- as_workflow_set(
  "boosted_tree" = boost_tune,
  "random_forest" = rf_tune,
  "elastic_net" = en_tune,
  "knn" = knn_tune,
  "mlp" = mlp_tune,
  "svm_poly" = svm_poly_tune,
  "svm_rad" = svm_rad_tune,
  "mars" = mars_tune
)

# plot results
model_set %>% 
  autoplot(metric = "roc_auc")

# plot just the best
model_set %>% 
  autoplot(metric = "roc_auc", select_best = TRUE) +
  theme_minimal() +
  geom_text(aes(y = mean - 0.03, label = wflow_id, angle = 90, hjust = 1)) +
  ylim(c(0.7, 0.9)) +
  theme(legend.position = "none")
# will want this in the report ! either save image or will need to include this code

# table of our results
model_results <- model_set %>% 
  group_by(wflow_id) %>% 
  mutate(best = map(result, show_best, metric = "roc_auc", n = 1)) %>% 
  unnest(cols = c(best))

# computation time
model_times <- bind_rows(boost_tictoc,
                         rf_tictoc,
                         en_tictoc,
                         knn_tictoc,
                         mlp_tictoc,
                         svm_poly_tictoc,
                         svm_rad_tictoc,
                         mars_tictoc) %>% 
  mutate(
    wflow_id = c(
      "boosted_tree",
      "random_forest",
      "elastic_net",
      "knn",
      "mlp",
      "svm_poly",
      "svm_rad",
      "mars"
    )
  )

result_table <- merge(model_results, model_times) %>% 
  select(model, mean, runtime) %>% 
  rename(roc_auc = mean)

save(result_table, file = "results/result_table.rda")


####################################

# finalize workflow
mars_workflow_tuned <- mars_workflow %>% 
  finalize_workflow(select_best(mars_tune, metric = "roc_auc"))

# fit to training data
mars_fit <- fit(mars_workflow_tuned, wildfire_train)

saveRDS(mars_fit, "results/mars_fit.rds")

# predict the testing data
mars_cat <- predict(mars_fit, wildfire_test, type = "class")


final_result <- wildfire_test %>% 
  select(wlf) %>% 
  bind_cols(mars_cat)

accuracy(final_result, wlf, .pred_class)

final_result %>% 
  conf_mat(wlf, .pred_class) %>% 
  autoplot(type = "heatmap")















