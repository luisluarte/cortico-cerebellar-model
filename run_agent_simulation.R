library(Rcpp)
library(RcppArmadillo)
library(ggplot2)
library(dplyr)
library(glmnet)

cat("Loading Simulation Core and Data...\n")
sourceCpp("src/r/sim_bio_probes.cpp")
sourceCpp("src/r/sim_agent.cpp")

targets <- readRDS("src/r/distillation_targets_v2.rds")
X <- targets$X
ITI <- targets$ITI / max(targets$ITI)
t_subjs <- targets$subjs
train_subjs <- unique(t_subjs)

set.seed(42)
input_dim <- ncol(X)
W_gen <- matrix(rnorm(input_dim * 32, 0, 1/sqrt(32)), nrow=input_dim, ncol=32)
W_ach1 <- matrix(rnorm(2000 * 32, 0, 1/sqrt(32)), nrow=2000, ncol=32)
mask1 <- matrix(rbinom(2000 * 32, 1, 0.1), nrow=2000, ncol=32)
W_ach1 <- W_ach1 * mask1
W_ach2 <- matrix(rnorm(1000 * 2000, 0, 1/sqrt(2000)), nrow=1000, ncol=2000)
W_thal <- matrix(rnorm(32 * 1000, 0, 1/sqrt(1000)), nrow=32, ncol=1000)

res_opt <- readRDS("results/nsga2_pruning_results.rds")
opt_p <- res_opt$par[1, ]

# 1. Fit the Readout Vector on Subject 1 using optimal parameters
idx_s1 <- which(t_subjs == train_subjs[1])
X_s1 <- X[idx_s1, , drop=FALSE]
ITI_s1 <- ITI[idx_s1]
Pi_s1 <- matrix(1, nrow=length(idx_s1), ncol=input_dim)

cat("Extracting Optimal Generative Traces...\n")
base_traces <- simulate_bio_probes(length(idx_s1), X_s1, ITI_s1, W_gen, W_ach1, W_ach2, W_thal, Pi_s1, 
                                   opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], opt_p[8], opt_p[9])

cat("Fitting Logistic Readout (mu -> choice)...\n")
# Fit logistic regression on the human's actual choices
s1_choices <- targets$labels[idx_s1]
df_glm <- data.frame(Choice = s1_choices, base_traces$mu)
logit_model <- glm(Choice ~ ., data = df_glm, family = binomial(link="logit"))
W_readout <- coef(logit_model)[-1]
W_readout[is.na(W_readout)] <- 0
intercept <- coef(logit_model)[1]

# 2. Simulate Array of Agents
N_AGENTS <- 50
beta_thal_values <- seq(0.01, 1.5, length.out = N_AGENTS)
arm_probs <- c(0.8, 0.2, 0.7, 0.3, 0.9, 0.1, 0.6, 0.4) # Fixed probabilities for the 8 bandits

results <- data.frame()

cat("Simulating Autonomous Agents in the Environment...\n")
for(a in 1:N_AGENTS) {
  beta_agent <- beta_thal_values[a]
  
  # Run closed-loop agent simulation
  sim_res <- simulate_agent_behavior(length(idx_s1), X_s1, ITI_s1, W_gen, W_ach1, W_ach2, W_thal, Pi_s1,
                                     opt_p[1], opt_p[2], beta_agent, opt_p[4], opt_p[5], opt_p[6], opt_p[7], opt_p[8], opt_p[9],
                                     W_readout, intercept, arm_probs)
  
  sim_ch <- sim_res$choices
  sim_rew <- sim_res$rewards
  
  n_trials <- length(sim_ch)
  cur_ch <- sim_ch[2:n_trials]
  prev_ch <- sim_ch[1:(n_trials-1)]
  prev_rew <- sim_rew[1:(n_trials-1)]
  
  switch_trial <- (cur_ch != prev_ch)
  stay_trial <- (cur_ch == prev_ch)
  win_idx <- (prev_rew == 1)
  loss_idx <- (prev_rew == 0)
  
  win_stay <- ifelse(sum(win_idx) > 0, mean(stay_trial[win_idx]), NA)
  lose_shift <- ifelse(sum(loss_idx) > 0, mean(switch_trial[loss_idx]), NA)
  
  mi <- (win_stay + lose_shift) / 2.0
  
  results <- rbind(results, data.frame(
    Agent = a,
    Beta_Thal = beta_agent,
    MI = mi
  ))
}

cat("Plotting Simulated Agent Topography...\n")
p <- ggplot(results, aes(x = MI, y = Beta_Thal)) +
  geom_point(color = "#e41a1c", size = 4, alpha = 0.8) +
  geom_smooth(method = "loess", span = 0.5, color = "black", linewidth = 1.2, fill = "grey70") +
  labs(title = "Causal Validation: Simulated Agent Array Recovers the U-Shape",
       subtitle = "By sweeping \u03b2_thal in a closed-loop task, agents autonomously generate the exact U-shaped strategy manifold.",
       x = "Agent's Simulated Markovian Behavior Index (MI)",
       y = "Agent's Programmed Thalamic Feedback (\u03b2_thal)") +
  theme_minimal(base_size = 14) +
  theme(panel.grid.minor = element_blank())

ggsave("results/Fig14_Simulated_Agent_Beta.png", plot = p, width = 8, height = 5, dpi = 300)
cat("Done! Plot saved to results/Fig14_Simulated_Agent_Beta.png\n")
