res <- readRDS('results/nsga2_pruning_results.rds')
df <- data.frame(
  Loss = res$value[,1],
  Granule = round(res$par[,8] * 2000),
  DCN = round(res$par[,9] * 1000)
)
write.csv(df, 'results/pruning_final_pop.csv', row.names=FALSE)
