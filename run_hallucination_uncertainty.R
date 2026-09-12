library(Rcpp)
library(RcppArmadillo)
library(glmnet)
library(ggplot2)
library(dplyr)
library(tidyr)
library(zoo)

cat("Loading Simulation Core and Data...\n")
sourceCpp("src/r/sim_bio_probes.cpp")
sourceCpp("src/r/sim_hallucination.cpp")

targets <- readRDS("src/r/distillation_targets_v2.rds")
X <- targets$X
ITI <- targets$ITI
ITI <- ITI / max(ITI)  # Normalized ITI
lag_Reward <- targets$lag_reward
t_subjs <- targets$subjs

set.seed(42)
input_dim <- ncol(X)
W_gen <- matrix(rnorm(input_dim * 32, 0, 1/sqrt(32)), nrow=input_dim, ncol=32)
W_ach1 <- matrix(rnorm(2000 * 32, 0, 1/sqrt(32)), nrow=2000, ncol=32)
mask1 <- matrix(rbinom(2000 * 32, 1, 0.1), nrow=2000, ncol=32)
W_ach1 <- W_ach1 * mask1
W_ach2 <- matrix(rnorm(1000 * 2000, 0, 1/sqrt(2000)), nrow=1000, ncol=2000)
W_thal <- matrix(rnorm(32 * 1000, 0, 1/sqrt(1000)), nrow=32, ncol=1000)

train_subjs <- unique(t_subjs)
res <- readRDS("results/nsga2_pruning_results.rds")
opt_p <- res$par[1, ]

df_plot <- data.frame()
n_steps <- 50

cat("Running Uncertainty Attractor Experiment over 10 Participants...\n")

for(s in 1:10) {
  cat(sprintf("Processing Participant %d / 10...\n", s))
  idx <- which(t_subjs == train_subjs[s])
  subj_X <- X[idx, , drop=FALSE]
  subj_ITI <- ITI[idx]
  subj_Pi <- matrix(1, nrow=length(idx), ncol=input_dim)
  subj_reward <- lag_Reward[idx]

  traces <- simulate_bio_probes(length(idx), subj_X, subj_ITI, W_gen, W_ach1, W_ach2, W_thal, subj_Pi, 
                                opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7],
                                opt_p[8], opt_p[9])

  # Compute Uncertainty (Reward Entropy)
  roll_win <- rollmean(subj_reward, k=10, fill=NA, align="right")
  eps_val <- 1e-5
  roll_win <- pmax(eps_val, pmin(1-eps_val, roll_win))
  unc_target <- -(roll_win * log2(roll_win) + (1-roll_win) * log2(1-roll_win))
  
  valid_idx <- which(!is.na(unc_target))
  unc_valid <- unc_target[valid_idx]
  # Min-Max Normalization to [0, 1]
  unc_norm <- (unc_valid - min(unc_valid)) / (max(unc_valid) - min(unc_valid) + 1e-9)

  X_mu <- traces$mu[valid_idx, ]
  cv_fit_mu <- cv.glmnet(X_mu, unc_norm, alpha = 0, nfolds = 5)

  high_trial <- valid_idx[which.max(unc_norm)]
  low_trial <- valid_idx[which.min(unc_norm)]

  # --- ITI = 1.0 (Short Rest) ---
  hal_high_1 <- simulate_hallucination(high_trial - 1, n_steps, 1.0, subj_X, subj_ITI, W_gen, W_ach1, W_ach2, W_thal, subj_Pi,
                                     opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], opt_p[8], opt_p[9])
  hal_low_1 <- simulate_hallucination(low_trial - 1, n_steps, 1.0, subj_X, subj_ITI, W_gen, W_ach1, W_ach2, W_thal, subj_Pi,
                                    opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], opt_p[8], opt_p[9])

  # --- ITI = 10.0 (Deep Rest / Sleep) ---
  hal_high_10 <- simulate_hallucination(high_trial - 1, n_steps, 10.0, subj_X, subj_ITI, W_gen, W_ach1, W_ach2, W_thal, subj_Pi,
                                     opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], opt_p[8], opt_p[9])
  hal_low_10 <- simulate_hallucination(low_trial - 1, n_steps, 10.0, subj_X, subj_ITI, W_gen, W_ach1, W_ach2, W_thal, subj_Pi,
                                    opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], opt_p[8], opt_p[9])

  # Decode
  unc_hh_1 <- predict(cv_fit_mu, newx = hal_high_1$mu_hallucinated, s = "lambda.min")
  unc_hl_1 <- predict(cv_fit_mu, newx = hal_low_1$mu_hallucinated, s = "lambda.min")
  unc_hh_10 <- predict(cv_fit_mu, newx = hal_high_10$mu_hallucinated, s = "lambda.min")
  unc_hl_10 <- predict(cv_fit_mu, newx = hal_low_10$mu_hallucinated, s = "lambda.min")

  # Store
  df_plot <- rbind(df_plot, data.frame(
    Subject = s,
    Timestep = rep(1:n_steps, 4),
    Attractor = c(rep("High-Uncertainty", n_steps), rep("Low-Uncertainty", n_steps), rep("High-Uncertainty", n_steps), rep("Low-Uncertainty", n_steps)),
    ITI = c(rep("ITI = 1.0 (Short Rest)", n_steps*2), rep("ITI = 10.0 (Deep Rest)", n_steps*2)),
    Decoded_Uncertainty = c(as.numeric(unc_hh_1), as.numeric(unc_hl_1), as.numeric(unc_hh_10), as.numeric(unc_hl_10))
  ))
}

cat("Plotting Aggregated Trajectories for Uncertainty...\n")

df_plot$Group <- paste(df_plot$Attractor, df_plot$ITI)

p <- ggplot(df_plot, aes(x = Timestep, y = Decoded_Uncertainty, color = Attractor, fill = Attractor, linetype = ITI)) +
  stat_summary(aes(group = Group), geom = "line", fun = mean, linewidth = 1.2) +
  stat_summary(aes(group = Group), geom = "ribbon", fun.data = mean_se, alpha = 0.15, color = NA) +
  coord_cartesian(ylim = c(0, 1)) +
  scale_y_continuous(breaks = seq(0, 1, 0.2)) +
  scale_color_manual(values = c("High-Uncertainty" = "#984ea3", "Low-Uncertainty" = "#ff7f00")) +
  scale_fill_manual(values = c("High-Uncertainty" = "#984ea3", "Low-Uncertainty" = "#ff7f00")) +
  labs(title = "Time-Dependent Uncertainty Decay During Reverberation",
       subtitle = "Cortical decay and Purkinje drift scaled by Rest Duration. Ribbons = ±1 SE (N=10).",
       x = "Autonomous Reverberation Timesteps (Sensory Deprivation)",
       y = "Decoded Reward Entropy (Normalized)") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "right",
        legend.box = "vertical",
        legend.title = element_text(face="bold"),
        panel.grid.minor = element_blank())

ggsave("results/Fig6_Hallucination_Uncertainty.png", plot = p, width = 9, height = 5, dpi = 300)
cat("Done! Plot saved to results/Fig6_Hallucination_Uncertainty.png\n")
