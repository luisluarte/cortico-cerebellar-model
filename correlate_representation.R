library(Rcpp)
library(RcppArmadillo)
library(ggplot2)
library(dplyr)
library(stats)

cat("Loading Data and Core...\n")
sourceCpp("src/r/sim_bio_probes.cpp")

targets <- readRDS("src/r/distillation_targets_v2.rds")
post_df <- readRDS("final_model/results/factorial_fixed_10k_A.rds")

X <- targets$X
ITI <- targets$ITI / max(targets$ITI)
rnn_mu <- targets$rnn_mu
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

results <- data.frame()

cat("Measuring Cortical Representation Quality for 50 Subjects...\n")
for(s in 1:length(train_subjs)) {
  idx <- which(t_subjs == train_subjs[s])
  if (length(idx) < 40) next # Skip if too few trials to fit 32 dims
  
  X_s <- X[idx, , drop=FALSE]
  ITI_s <- ITI[idx]
  Pi_s <- matrix(1, nrow=length(idx), ncol=input_dim)
  
  p <- params_exp[s, ]
  
  traces <- simulate_bio_probes(length(idx), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s, 
                                p[1], p[2], p[3], p[4], p[5], p[6], p[7], 0.1, 0.1)
  
  mu_bio <- traces$mu
  mu_target <- rnn_mu[idx, , drop=FALSE]
  
  # Measure how well the 32-dim biological cortex can linearly represent the first 3 dims of the target state
  r2_sum <- 0
  for(c in 1:3) {
    df <- data.frame(y = mu_target[, c], mu_bio)
    fit <- lm(y ~ ., data=df)
    r2_sum <- r2_sum + summary(fit)$r.squared
  }
  mean_r2 <- r2_sum / 3
  
  results <- rbind(results, data.frame(
    Subject = s,
    Beta_Thal = p[3],
    Cortical_R2 = mean_r2
  ))
}

test <- cor.test(results$Beta_Thal, results$Cortical_R2, method="spearman")
cat(sprintf("Correlation Beta_Thal vs Cortical R2: rho = %.3f, p = %.3e\n", test$estimate, test$p.value))

p <- ggplot(results, aes(x = Beta_Thal, y = Cortical_R2)) +
  geom_point(color = "#984ea3", size = 4, alpha = 0.8) +
  geom_smooth(method = "lm", color = "black", linewidth = 1.2) +
  labs(title = "Cortical Representation Capacity vs Thalamic Rigidity",
       subtitle = sprintf("How well the Cortex encodes task states given \u03b2_thal (\u03c1 = %.3f, p = %.3f).", test$estimate, test$p.value),
       x = "Thalamic Feedback Rigidity (\u03b2_thal)",
       y = "Cortical Representation Capacity (Linear R\u00b2)") +
  theme_minimal(base_size = 14)

ggsave("results/Fig18_Representation_Beta.png", plot = p, width = 8, height = 5, dpi = 300)
cat("Done! Plot saved to results/Fig18_Representation_Beta.png\n")
