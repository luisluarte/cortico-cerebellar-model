library(Rcpp)
library(RcppArmadillo)
library(ggplot2)
library(dplyr)
library(glmnet)

cat("Loading Simulation Core and Data...\n")
sourceCpp("src/r/sim_bio_probes.cpp")
sourceCpp("src/r/sim_restless_agent.cpp")

targets <- readRDS("src/r/distillation_targets_v2.rds")
post_df <- readRDS("final_model/results/factorial_fixed_10k_A.rds")

X <- targets$X
ITI <- targets$ITI / max(targets$ITI)
input_dim <- ncol(X)

set.seed(42)
W_gen <- matrix(rnorm(input_dim * 32, 0, 1/sqrt(32)), nrow=input_dim, ncol=32)
W_ach1 <- matrix(rnorm(2000 * 32, 0, 1/sqrt(32)), nrow=2000, ncol=32)
mask1 <- matrix(rbinom(2000 * 32, 1, 0.1), nrow=2000, ncol=32)
W_ach1 <- W_ach1 * mask1
W_ach2 <- matrix(rnorm(1000 * 2000, 0, 1/sqrt(2000)), nrow=1000, ncol=2000)
W_thal <- matrix(rnorm(32 * 1000, 0, 1/sqrt(1000)), nrow=32, ncol=1000)

res_opt <- readRDS("results/nsga2_pruning_results.rds")
opt_p <- res_opt$par[1, ]

# 1. Fit W_readout using optimal parameters on Subject 1 (to ensure generic choice mapping)
idx_s1 <- which(targets$subjs == unique(targets$subjs)[1])
base_traces <- simulate_bio_probes(length(idx_s1), X[idx_s1, , drop=FALSE], ITI[idx_s1], 
                                   W_gen, W_ach1, W_ach2, W_thal, matrix(1, nrow=length(idx_s1), ncol=input_dim), 
                                   opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], opt_p[8], opt_p[9])

logit_model <- glm(targets$labels[idx_s1] ~ ., data = data.frame(Choice = targets$labels[idx_s1], base_traces$mu), family = binomial(link="logit"))
raw_coef <- coef(logit_model)
intercept <- raw_coef[1]
if(is.na(intercept)) intercept <- 0
W_readout <- rep(0, 32)
for(i in 1:32) {
  name <- paste0("X", i) # base_traces$mu columns become X1..X32 by default in data.frame if unnamed
  if(name %in% names(raw_coef)) {
    W_readout[i] <- raw_coef[name]
  }
}
W_readout[is.na(W_readout)] <- 0

# 2. Extract 10 random Human Parameter bases
emp_params <- t(sapply(post_df[['bio_dist']][['traces']], function(x) apply(x, 3, mean)))
params_exp <- exp(emp_params)
sampled_idx <- sample(1:nrow(params_exp), 10, replace = TRUE)
base_agents <- params_exp[sampled_idx, ]

# 3. Sweep Grid
N_TRIALS <- 1000
drifts <- c(0.01, 0.05, 0.15)
drift_labels <- c("Low Drift (Highly Non-Markovian)", "Medium Drift", "High Drift (Highly Markovian)")
beta_sweep <- seq(0.05, 1.5, length.out = 15)

results <- data.frame()

cat("Running Massive Simulation Sweep...\n")
for(d in 1:length(drifts)) {
  for(a in 1:nrow(base_agents)) {
    agent_p <- base_agents[a, ]
    
    for(b in 1:length(beta_sweep)) {
      b_val <- beta_sweep[b]
      
      # Agent uses its own parameters, but beta_thal is overridden by the sweep
      sim_res <- sim_restless_agent(N_TRIALS, drifts[d], W_gen, W_ach1, W_ach2, W_thal, rep(1, input_dim),
                                    agent_p[1], agent_p[2], b_val, agent_p[4], agent_p[5], agent_p[6], agent_p[7], opt_p[8], opt_p[9],
                                    W_readout, intercept)
      
      sim_ch <- sim_res$choices
      sim_rew <- sim_res$rewards
      
      cur_ch <- sim_ch[2:N_TRIALS]
      prev_ch <- sim_ch[1:(N_TRIALS-1)]
      prev_rew <- sim_rew[1:(N_TRIALS-1)]
      
      switch_trial <- (cur_ch != prev_ch)
      stay_trial <- (cur_ch == prev_ch)
      win_idx <- (prev_rew == 1)
      loss_idx <- (prev_rew == 0)
      
      win_stay <- ifelse(sum(win_idx) > 0, mean(stay_trial[win_idx]), NA)
      lose_shift <- ifelse(sum(loss_idx) > 0, mean(switch_trial[loss_idx]), NA)
      
      mi <- (win_stay + lose_shift) / 2.0
      total_reward <- sum(sim_rew)
      
      results <- rbind(results, data.frame(
        Volatility = drift_labels[d],
        Base_Agent = factor(a),
        Beta_Thal = b_val,
        MI = mi,
        Cumulative_Reward = total_reward
      ))
    }
  }
}

# Preserve ordering of Volatility factor
results$Volatility <- factor(results$Volatility, levels = drift_labels)

cat("Plotting MI vs Beta_Thal...\n")
p1 <- ggplot(results, aes(x = Beta_Thal, y = MI, color = Base_Agent)) +
  geom_line(alpha = 0.5) +
  geom_smooth(aes(group = 1), method = "loess", color = "black", linewidth = 1.2, se = FALSE) +
  facet_wrap(~Volatility) +
  labs(title = "Environmental Volatility Drives the Simulated Markovian U-Shape",
       subtitle = "10 simulated agents tested across varying Restless Bandit non-Markovian drift rates.",
       x = "Swept Thalamic Feedback (\u03b2_thal)",
       y = "Simulated Markovian Index (MI)") +
  theme_minimal(base_size = 14) + theme(legend.position = "none")

ggsave("results/Fig15A_Restless_MI.png", plot = p1, width = 11, height = 5, dpi = 300)

cat("Plotting Cumulative Reward vs Beta_Thal...\n")
p2 <- ggplot(results, aes(x = Beta_Thal, y = Cumulative_Reward, color = Base_Agent)) +
  geom_line(alpha = 0.5) +
  geom_smooth(aes(group = 1), method = "loess", color = "black", linewidth = 1.2, se = FALSE) +
  facet_wrap(~Volatility) +
  labs(title = "Optimal \u03b2_thal Shifts According to Environmental Volatility",
       subtitle = "High drift strictly demands highly reactive (low \u03b2) attractors to maximize performance.",
       x = "Swept Thalamic Feedback (\u03b2_thal)",
       y = "Cumulative Reward (1000 trials)") +
  theme_minimal(base_size = 14) + theme(legend.position = "none")

ggsave("results/Fig15B_Restless_Reward.png", plot = p2, width = 11, height = 5, dpi = 300)
cat("Done! Saved Fig15A and Fig15B.\n")
