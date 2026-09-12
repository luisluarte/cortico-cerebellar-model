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

cat("Fetching Optimal God Parameters...\n")
res <- readRDS("results/nsga2_pruning_results.rds")
opt_p <- res$par[1, ]

df_plot <- data.frame()
n_steps <- 50

cat("Running Experiment over 10 Participants...\n")

for(s in 1:10) {
  cat(sprintf("Processing Participant %d / 10...\n", s))
  
  idx <- which(t_subjs == train_subjs[s])
  subj_X <- X[idx, , drop=FALSE]
  subj_ITI <- ITI[idx]
  subj_Pi <- matrix(1, nrow=length(idx), ncol=input_dim)
  subj_reward <- lag_Reward[idx]

  # 1. Normal Trial Extraction
  traces <- simulate_bio_probes(length(idx), subj_X, subj_ITI, W_gen, W_ach1, W_ach2, W_thal, subj_Pi, 
                                opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7],
                                opt_p[8], opt_p[9])

  # Compute and Normalize Value
  val_target <- rollmean(subj_reward, k=10, fill=NA, align="right")
  valid_idx <- which(!is.na(val_target))
  
  val_valid <- val_target[valid_idx]
  # Min-Max Normalization to [0, 1]
  val_norm <- (val_valid - min(val_valid)) / (max(val_valid) - min(val_valid) + 1e-9)

  # 2. Train Value Decoder
  X_mu <- traces$mu[valid_idx, ]
  cv_fit_mu <- cv.glmnet(X_mu, val_norm, alpha = 0, nfolds = 5)

  # 3. Find Target Cutoff Trials
  high_trial <- valid_idx[which.max(val_norm)]
  low_trial <- valid_idx[which.min(val_norm)]

  # 4. Run Hallucination
  # Sensory Deprivation: Implicitly uses ITI = 1.0 (max normalized rest period) per timestep in C++
  hal_high <- simulate_hallucination(high_trial - 1, n_steps, subj_X, subj_ITI, W_gen, W_ach1, W_ach2, W_thal, subj_Pi,
                                     opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], opt_p[8], opt_p[9])

  hal_low <- simulate_hallucination(low_trial - 1, n_steps, subj_X, subj_ITI, W_gen, W_ach1, W_ach2, W_thal, subj_Pi,
                                    opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], opt_p[8], opt_p[9])

  # 5. Decode Hallucinated State
  val_high_hallucinated <- predict(cv_fit_mu, newx = hal_high$mu_hallucinated, s = "lambda.min")
  val_low_hallucinated <- predict(cv_fit_mu, newx = hal_low$mu_hallucinated, s = "lambda.min")

  # 6. Store
  df_subj <- data.frame(
    Subject = s,
    Timestep = rep(1:n_steps, 2),
    State = c(rep("High-Value Attractor", n_steps), rep("Low-Value Attractor", n_steps)),
    Decoded_Value = c(as.numeric(val_high_hallucinated), as.numeric(val_low_hallucinated))
  )
  df_plot <- rbind(df_plot, df_subj)
}

cat("Plotting Aggregated Trajectories...\n")
p <- ggplot(df_plot, aes(x = Timestep, y = Decoded_Value, color = State, fill = State)) +
  stat_summary(geom = "line", fun = mean, linewidth = 1.5) +
  stat_summary(geom = "ribbon", fun.data = mean_se, alpha = 0.2, color = NA) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  scale_color_manual(values = c("High-Value Attractor" = "#2a9d7a", "Low-Value Attractor" = "#d95f02")) +
  scale_fill_manual(values = c("High-Value Attractor" = "#2a9d7a", "Low-Value Attractor" = "#d95f02")) +
  labs(title = "Generative Replay: Stable Value Attractors across 10 Participants",
       subtitle = "Values min-max normalized [0, 1]. Ribbons represent ±1 SE across N=10.",
       x = "Autonomous Reverberation Timesteps (Sensory Deprivation)",
       y = "Decoded Expected Value (Normalized)") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom",
        legend.title = element_blank(),
        panel.grid.minor = element_blank())

ggsave("results/Fig6_Hallucination_10Subjs.png", plot = p, width = 8, height = 5, dpi = 300)
cat("Done! Plot saved to results/Fig6_Hallucination_10Subjs.png\n")
