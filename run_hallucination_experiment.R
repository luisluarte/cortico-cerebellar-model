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
ITI <- ITI / max(ITI)
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
idx <- which(t_subjs == train_subjs[1])
subj_X <- X[idx, , drop=FALSE]
subj_ITI <- ITI[idx]
subj_Pi <- matrix(1, nrow=length(idx), ncol=input_dim)
subj_reward <- lag_Reward[idx]

cat("Fetching Optimal God Parameters...\n")
res <- readRDS("results/nsga2_pruning_results.rds")
opt_p <- res$par[1, ]

cat("Step 1: Running Normal Trial Extraction...\n")
traces <- simulate_bio_probes(length(idx), subj_X, subj_ITI, W_gen, W_ach1, W_ach2, W_thal, subj_Pi, 
                              opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7],
                              opt_p[8], opt_p[9])

val_target <- rollmean(subj_reward, k=10, fill=NA, align="right")
valid_idx <- which(!is.na(val_target))

cat("Step 2: Training Value Decoder on Cortical Trace...\n")
X_mu <- traces$mu[valid_idx, ]
y_val <- val_target[valid_idx]

cv_fit_mu <- cv.glmnet(X_mu, y_val, alpha = 0, nfolds = 5)

cat("Step 3: Finding Target Cutoff Trials...\n")
# Find a trial where expected value is HIGH
high_trial <- valid_idx[which.max(y_val)]
# Find a trial where expected value is LOW
low_trial <- valid_idx[which.min(y_val)]

cat(sprintf("High Value Trial Cutoff: %d\n", high_trial))
cat(sprintf("Low Value Trial Cutoff: %d\n", low_trial))

cat("Step 4: Running Hallucination Simulation (Sensory Deprivation)...\n")
n_steps <- 50

hal_high <- simulate_hallucination(high_trial, n_steps, subj_X, subj_ITI, W_gen, W_ach1, W_ach2, W_thal, subj_Pi,
                                   opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], opt_p[8], opt_p[9])

hal_low <- simulate_hallucination(low_trial, n_steps, subj_X, subj_ITI, W_gen, W_ach1, W_ach2, W_thal, subj_Pi,
                                  opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], opt_p[8], opt_p[9])

cat("Step 5: Decoding Hallucinated Mind-State...\n")
val_high_hallucinated <- predict(cv_fit_mu, newx = hal_high$mu_hallucinated, s = "lambda.min")
val_low_hallucinated <- predict(cv_fit_mu, newx = hal_low$mu_hallucinated, s = "lambda.min")

df_plot <- data.frame(
  Timestep = rep(1:n_steps, 2),
  State = c(rep("Hallucinating High-Value State", n_steps), rep("Hallucinating Low-Value State", n_steps)),
  Decoded_Value = c(as.numeric(val_high_hallucinated), as.numeric(val_low_hallucinated))
)

p <- ggplot(df_plot, aes(x = Timestep, y = Decoded_Value, color = State)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = c("Hallucinating High-Value State" = "#2a9d7a", "Hallucinating Low-Value State" = "#d95f02")) +
  labs(title = "Decoding the Hallucinated Value State During Sensory Deprivation",
       subtitle = "Cortico-Cerebellar loop runs autonomously (X=0). Value decoded from Cortical reverberation.",
       x = "Autonomous Timesteps (ITI Dreaming)",
       y = "Decoded Expected Value") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom",
        legend.title = element_blank(),
        panel.grid.minor = element_blank())

ggsave("results/Fig6_Hallucination_Attractors.png", plot = p, width = 8, height = 5, dpi = 300)
cat("Done! Plot saved to results/Fig6_Hallucination_Attractors.png\n")
