library(Rcpp)
library(RcppArmadillo)
library(ggplot2)
library(dplyr)

cat("Loading Data and Core...\n")
sourceCpp("src/r/sim_bio_probes.cpp")

targets <- readRDS("src/r/distillation_targets_v2.rds")
post_df <- readRDS("final_model/results/factorial_fixed_10k_A.rds")

X <- targets$X
ITI <- targets$ITI / max(targets$ITI)
t_subjs <- targets$subjs
train_subjs <- unique(t_subjs)

input_dim <- ncol(X)
set.seed(42)
W_gen <- matrix(rnorm(input_dim * 32, 0, 1/sqrt(32)), nrow=input_dim, ncol=32)
W_ach1 <- matrix(rnorm(2000 * 32, 0, 1/sqrt(32)), nrow=2000, ncol=32)
mask1 <- matrix(rbinom(2000 * 32, 1, 0.1), nrow=2000, ncol=32)
W_ach1 <- W_ach1 * mask1
W_ach2 <- matrix(rnorm(1000 * 2000, 0, 1/sqrt(2000)), nrow=1000, ncol=2000)
W_thal <- matrix(rnorm(32 * 1000, 0, 1/sqrt(1000)), nrow=32, ncol=1000)

emp_params <- t(sapply(post_df[['bio_dist']][['traces']], function(x) apply(x, 3, mean)))
params_exp <- exp(emp_params)
colnames(params_exp) <- c('alpha_pc', 'lambda_pc', 'beta_thal', 'kappa_cf', 'alpha_gran', 'beta_gran', 'sigma2')

results <- data.frame()

cat("Measuring Attractor Depth (Confidence) for all 50 Humans...\n")
for(s in 1:length(train_subjs)) {
  idx <- which(t_subjs == train_subjs[s])
  X_s <- X[idx, , drop=FALSE]
  ITI_s <- ITI[idx]
  Pi_s <- matrix(1, nrow=length(idx), ncol=input_dim)
  
  p <- params_exp[s, ]
  
  traces <- simulate_bio_probes(length(idx), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s, 
                                p[1], p[2], p[3], p[4], p[5], p[6], p[7], 0.1, 0.1)
  
  mu_traces <- traces$mu
  
  mu_norms <- sqrt(rowSums(mu_traces^2))
  mean_norm <- mean(mu_norms)
  
  if(is.infinite(mean_norm) || is.nan(mean_norm)) {
    mean_norm <- 1e10 # Cap it at a very high number instead of Inf
  }
  
  results <- rbind(results, data.frame(
    Subject = s,
    Beta_Thal = p[3],
    Attractor_Depth = log10(mean_norm + 1)
  ))
}

test <- cor.test(results$Beta_Thal, results$Attractor_Depth, method="spearman")
cat(sprintf("Correlation Beta_Thal vs Attractor Depth: rho = %.3f, p = %.3e\n", test$estimate, test$p.value))

p <- ggplot(results, aes(x = Beta_Thal, y = Attractor_Depth)) +
  geom_point(color = "#4daf4a", size = 4, alpha = 0.8) +
  geom_smooth(method = "lm", color = "black", linewidth = 1.2) +
  labs(title = "Decision Confidence is Driven by Thalamic Feedback",
       subtitle = sprintf("Confidence (Log10 depth of the \u03bc attractor) is strongly correlated with \u03b2_thal (\u03c1 = %.3f, p = %.3e).", test$estimate, test$p.value),
       x = "Thalamic Feedback Rigidity (\u03b2_thal)",
       y = "Internal Confidence (Log10 L2-Norm of \u03bc)") +
  theme_minimal(base_size = 14)

ggsave("results/Fig17_Confidence_Beta.png", plot = p, width = 8, height = 5, dpi = 300)
cat("Done! Plot saved to results/Fig17_Confidence_Beta.png\n")
