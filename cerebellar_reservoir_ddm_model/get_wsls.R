res <- readRDS('nsga2_wsls_results.rds')
vals <- res$value
pars <- res$par

sorted_idx <- order(vals[,1])
vals <- vals[sorted_idx, ]
pars <- pars[sorted_idx, ]

nll <- vals[,1]
gamma <- pars[,3]

total_nll_loss <- max(nll) - min(nll)
total_gamma_gain <- max(gamma) - min(gamma)
avg_cost <- total_nll_loss / total_gamma_gain

cat(sprintf("Total NLL Loss: %.4f\n", total_nll_loss))
cat(sprintf("Total Gamma Gain: %.4f\n", total_gamma_gain))
cat(sprintf("Average Cost (dNLL/dGamma): %.4f\n", avg_cost))
cat(sprintf("Optimal Theta Win at max Gamma: %.4f\n", pars[nrow(pars), 1]))
cat(sprintf("Optimal Theta Loss at max Gamma: %.4f\n", pars[nrow(pars), 2]))
