
extract_quartiles <- function(res_file, param_names) {
  res <- readRDS(res_file)
  mat <- res$par[res$pareto.optimal, , drop=FALSE]
  
  # Completely strip matrix attributes by coercing each column to pure numeric
  df <- as.data.frame(lapply(1:ncol(mat), function(i) as.numeric(mat[, i])))
  colnames(df) <- param_names
  
  df <- df[order(df$gamma), ]
  n <- nrow(df)
  q1 <- df[max(1, floor(n * 0.25)), ]
  q2 <- df[max(1, floor(n * 0.50)), ]
  q3 <- df[max(1, floor(n * 0.75)), ]
  
  out <- rbind(q1, q2, q3)
  out$Quartile <- c("Q1", "Q2", "Q3")
  
  # Reorder columns to put Quartile first
  out <- out[, c("Quartile", param_names)]
  rownames(out) <- NULL
  return(out)
}

wsls <- extract_quartiles("fast_res_wsls.rds", c("theta_win", "theta_loss", "gamma"))
qlearn <- extract_quartiles("fast_res_qlearn.rds", c("alpha_win", "alpha_loss", "beta", "gamma"))
bio <- extract_quartiles("fast_res_bio.rds", c("alpha_pc", "lambda_pc", "beta_thal", "kappa_cf", "alpha_gran", "beta_gran", "sigma2_diff", "gamma"))

param_stability <- list(
  Bio = bio,
  WSLS = wsls,
  QLearn = qlearn
)

saveRDS(param_stability, "parameter_stability.rds")
cat("Successfully generated perfectly flat parameter_stability.rds\n")
