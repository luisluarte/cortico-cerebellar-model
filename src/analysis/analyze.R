res <- readRDS('results/nsga2_pruning_results.rds')
df <- data.frame(
  Loss = res$value[,1],
  Cost = res$value[,2],
  Granule = round(res$par[,8] * 2000),
  DCN = round(res$par[,9] * 1000)
)
df <- df[order(df$Loss), ]
print(head(df, 15))
