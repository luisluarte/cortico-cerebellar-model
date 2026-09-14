targets <- readRDS("src/r/distillation_targets_v2.rds")
print(head(targets$lag_reward))
print(unique(targets$lag_reward))
