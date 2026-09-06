library(mco)

res <- readRDS("nsga2_moe_results.rds")



vals <- res$value
pars <- res$par

# Sort by Objective 1 (NLL)
sort_idx <- order(vals[, 1])
vals <- vals[sort_idx, , drop=FALSE]
pars <- pars[sort_idx, , drop=FALSE]

# Convert gamma_raw (param 8) to actual gamma
gamma_actual <- 1 / (1 + exp(-pars[, 8]))

# Print header
cat("NLL,Reliance,Gamma,alpha_pc,lambda_pc,beta_thal,kappa_cf,alpha_gran,beta_gran,sigma2_diff\n")
for(i in 1:nrow(vals)) {
  cat(sprintf("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", 
      vals[i,1], vals[i,2], gamma_actual[i], 
      pars[i,1], pars[i,2], pars[i,3], pars[i,4], pars[i,5], pars[i,6], pars[i,7]))
}
