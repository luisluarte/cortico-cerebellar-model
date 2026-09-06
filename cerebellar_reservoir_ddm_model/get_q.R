res <- readRDS('nsga2_qlearning_results.rds')
vals <- res$value
pars <- res$par

# The final population is the Pareto front approximation
# Sort by NLL
sorted_idx <- order(vals[,1])
vals <- vals[sorted_idx, ]
pars <- pars[sorted_idx, ]

# vals[,1] is NLL. vals[,2] is (1 - Gamma). So Gamma = pars[,4] = 1 - vals[,2].
nll <- vals[,1]
gamma <- pars[,4]

total_nll_loss <- max(nll) - min(nll)
total_gamma_gain <- max(gamma) - min(gamma)
avg_cost <- total_nll_loss / total_gamma_gain

cat(sprintf("Total NLL Loss: %.4f\n", total_nll_loss))
cat(sprintf("Total Gamma Gain: %.4f\n", total_gamma_gain))
cat(sprintf("Average Cost (dNLL/dGamma): %.4f\n", avg_cost))
