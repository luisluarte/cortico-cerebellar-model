
res <- readRDS("cortico-cerebellar-model/final_model/results/factorial_final_adaptive_checkpoint.rds")
for(m in names(res)) {
  cat("\n---", m, "---\n")
  cat("ELPD:", res[[m]]$loo$estimates["elpd_loo", "Estimate"], "\n")
  cat("Min Rhat:", min(res[[m]]$rhats, na.rm=TRUE), "\n")
  cat("Median Rhat:", median(res[[m]]$rhats, na.rm=TRUE), "\n")
  cat("Max Rhat:", max(res[[m]]$rhats, na.rm=TRUE), "\n")
}
