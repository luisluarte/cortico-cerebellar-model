res <- readRDS('nsga2_qlearning_results.rds')
vals <- res\$value
pars <- res\$par
p_idx <- mco::paretoFilter(vals)
# p_idx is boolean
if(sum(p_idx) == 1) {
  p_vals <- matrix(vals[p_idx, ], nrow=1)
  p_pars <- matrix(pars[p_idx, ], nrow=1)
} else {
  p_vals <- vals[p_idx, ]
  p_pars <- pars[p_idx, ]
}
sorted_idx <- order(p_vals[, 1])
p_vals <- p_vals[sorted_idx, , drop=FALSE]
p_pars <- p_pars[sorted_idx, , drop=FALSE]

out_df <- data.frame(
  NLL = p_vals[, 1],
  RNN_Reliance = p_vals[, 2],
  Gamma = p_pars[, 4],
  Alpha_Win = p_pars[, 1],
  Alpha_Loss = p_pars[, 2],
  Beta = p_pars[, 3]
)
write.csv(out_df, "qlearning_pareto_fixed.csv", row.names=FALSE)
