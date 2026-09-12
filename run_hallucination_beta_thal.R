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
res <- readRDS("results/nsga2_pruning_results.rds")
opt_p <- res$par[1, ]

opt_beta <- opt_p[3]
beta_vals <- c(0.0, opt_beta, 0.99)
beta_labels <- c("1. Lesioned (\u03b2 = 0.0)", 
                 sprintf("2. Optimal (\u03b2 = %.2f)", opt_beta), 
                 "3. Hyperactive (\u03b2 = 0.99)")

df_plot <- data.frame()
n_steps <- 50
N_SUBJECTS <- 50

cat(sprintf("Running Beta_Thal Experiment over %d Participants...\n", N_SUBJECTS))

for(s in 1:N_SUBJECTS) {
  cat(sprintf("Processing Participant %d / %d...\n", s, N_SUBJECTS))
  idx <- which(t_subjs == train_subjs[s])
  subj_X <- X[idx, , drop=FALSE]
  subj_ITI <- ITI[idx]
  subj_Pi <- matrix(1, nrow=length(idx), ncol=input_dim)
  subj_reward <- lag_Reward[idx]

  # 1. Base Forward Pass (always uses optimal beta to train the pure decoder)
  traces <- simulate_bio_probes(length(idx), subj_X, subj_ITI, W_gen, W_ach1, W_ach2, W_thal, subj_Pi, 
                                opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7],
                                opt_p[8], opt_p[9])

  # Targets
  val_target <- rollmean(subj_reward, k=10, fill=NA, align="right")
  roll_win <- pmax(1e-5, pmin(1-1e-5, val_target))
  unc_target <- -(roll_win * log2(roll_win) + (1-roll_win) * log2(1-roll_win))
  
  valid_idx <- which(!is.na(val_target) & !is.na(unc_target))
  val_v <- val_target[valid_idx]
  unc_v <- unc_target[valid_idx]
  
  val_norm <- (val_v - min(val_v)) / (max(val_v) - min(val_v) + 1e-9)
  unc_norm <- (unc_v - min(unc_v)) / (max(unc_v) - min(unc_v) + 1e-9)

  X_mu <- traces$mu[valid_idx, ]
  
  # Prevent constant target crashes
  if(var(val_norm) < 1e-6 || var(unc_norm) < 1e-6) next
  
  cv_fit_val <- cv.glmnet(X_mu, val_norm, alpha = 0, nfolds = 5)
  cv_fit_unc <- cv.glmnet(X_mu, unc_norm, alpha = 0, nfolds = 5)

  h_val <- valid_idx[which.max(val_norm)]
  l_val <- valid_idx[which.min(val_norm)]
  h_unc <- valid_idx[which.max(unc_norm)]
  l_unc <- valid_idx[which.min(unc_norm)]

  for(b in 1:length(beta_vals)) {
    curr_beta <- beta_vals[b]
    b_label <- beta_labels[b]
    
    # VALUE HALLUCINATIONS
    h_v_1 <- simulate_hallucination(h_val - 1, n_steps, 1.0, subj_X, subj_ITI, W_gen, W_ach1, W_ach2, W_thal, subj_Pi,
                                    opt_p[1], opt_p[2], curr_beta, opt_p[4], opt_p[5], opt_p[6], opt_p[7], opt_p[8], opt_p[9])
    l_v_1 <- simulate_hallucination(l_val - 1, n_steps, 1.0, subj_X, subj_ITI, W_gen, W_ach1, W_ach2, W_thal, subj_Pi,
                                    opt_p[1], opt_p[2], curr_beta, opt_p[4], opt_p[5], opt_p[6], opt_p[7], opt_p[8], opt_p[9])
    h_v_10 <- simulate_hallucination(h_val - 1, n_steps, 10.0, subj_X, subj_ITI, W_gen, W_ach1, W_ach2, W_thal, subj_Pi,
                                     opt_p[1], opt_p[2], curr_beta, opt_p[4], opt_p[5], opt_p[6], opt_p[7], opt_p[8], opt_p[9])
    l_v_10 <- simulate_hallucination(l_val - 1, n_steps, 10.0, subj_X, subj_ITI, W_gen, W_ach1, W_ach2, W_thal, subj_Pi,
                                     opt_p[1], opt_p[2], curr_beta, opt_p[4], opt_p[5], opt_p[6], opt_p[7], opt_p[8], opt_p[9])

    # UNCERTAINTY HALLUCINATIONS
    h_u_1 <- simulate_hallucination(h_unc - 1, n_steps, 1.0, subj_X, subj_ITI, W_gen, W_ach1, W_ach2, W_thal, subj_Pi,
                                    opt_p[1], opt_p[2], curr_beta, opt_p[4], opt_p[5], opt_p[6], opt_p[7], opt_p[8], opt_p[9])
    l_u_1 <- simulate_hallucination(l_unc - 1, n_steps, 1.0, subj_X, subj_ITI, W_gen, W_ach1, W_ach2, W_thal, subj_Pi,
                                    opt_p[1], opt_p[2], curr_beta, opt_p[4], opt_p[5], opt_p[6], opt_p[7], opt_p[8], opt_p[9])
    h_u_10 <- simulate_hallucination(h_unc - 1, n_steps, 10.0, subj_X, subj_ITI, W_gen, W_ach1, W_ach2, W_thal, subj_Pi,
                                     opt_p[1], opt_p[2], curr_beta, opt_p[4], opt_p[5], opt_p[6], opt_p[7], opt_p[8], opt_p[9])
    l_u_10 <- simulate_hallucination(l_unc - 1, n_steps, 10.0, subj_X, subj_ITI, W_gen, W_ach1, W_ach2, W_thal, subj_Pi,
                                     opt_p[1], opt_p[2], curr_beta, opt_p[4], opt_p[5], opt_p[6], opt_p[7], opt_p[8], opt_p[9])

    # Decode Value
    dv_hh_1 <- predict(cv_fit_val, newx = h_v_1$mu_hallucinated, s = "lambda.min")
    dv_hl_1 <- predict(cv_fit_val, newx = l_v_1$mu_hallucinated, s = "lambda.min")
    dv_hh_10 <- predict(cv_fit_val, newx = h_v_10$mu_hallucinated, s = "lambda.min")
    dv_hl_10 <- predict(cv_fit_val, newx = l_v_10$mu_hallucinated, s = "lambda.min")

    # Decode Uncertainty
    du_hh_1 <- predict(cv_fit_unc, newx = h_u_1$mu_hallucinated, s = "lambda.min")
    du_hl_1 <- predict(cv_fit_unc, newx = l_u_1$mu_hallucinated, s = "lambda.min")
    du_hh_10 <- predict(cv_fit_unc, newx = h_u_10$mu_hallucinated, s = "lambda.min")
    du_hl_10 <- predict(cv_fit_unc, newx = l_u_10$mu_hallucinated, s = "lambda.min")

    df_plot <- rbind(df_plot, data.frame(
      Subject = s, Timestep = rep(1:n_steps, 8),
      Metric = c(rep("Expected Value", n_steps*4), rep("Reward Entropy", n_steps*4)),
      Beta_Condition = b_label,
      Attractor = c(rep("High", n_steps), rep("Low", n_steps), rep("High", n_steps), rep("Low", n_steps),
                    rep("High", n_steps), rep("Low", n_steps), rep("High", n_steps), rep("Low", n_steps)),
      ITI = c(rep("ITI = 1.0", n_steps*2), rep("ITI = 10.0", n_steps*2), rep("ITI = 1.0", n_steps*2), rep("ITI = 10.0", n_steps*2)),
      Decoded_Score = c(as.numeric(dv_hh_1), as.numeric(dv_hl_1), as.numeric(dv_hh_10), as.numeric(dv_hl_10),
                        as.numeric(du_hh_1), as.numeric(du_hl_1), as.numeric(du_hh_10), as.numeric(du_hl_10))
    ))
  }
}

cat("Aggregating and Plotting Massive Facet Grid...\n")

df_plot$Group <- paste(df_plot$Attractor, df_plot$ITI)

# Create a clean factor for facet ordering
df_plot$Metric <- factor(df_plot$Metric, levels = c("Expected Value", "Reward Entropy"))

p <- ggplot(df_plot, aes(x = Timestep, y = Decoded_Score, color = Attractor, fill = Attractor, linetype = ITI)) +
  stat_summary(aes(group = Group), geom = "line", fun = mean, linewidth = 1.2) +
  stat_summary(aes(group = Group), geom = "ribbon", fun.data = mean_se, alpha = 0.2, color = NA) +
  facet_grid(Metric ~ Beta_Condition, scales = "free_y") +
  coord_cartesian(ylim = c(0, 1.2)) + 
  scale_color_manual(values = c("High" = "#984ea3", "Low" = "#ff7f00")) +
  scale_fill_manual(values = c("High" = "#984ea3", "Low" = "#ff7f00")) +
  labs(title = "Differential Cortico-Cerebellar Dynamics Driven by Thalamic Feedback",
       subtitle = "Attractor stability (Value and Uncertainty) critically depends on the loop's feedback parameter (\u03b2_thal). N=50.",
       x = "Autonomous Reverberation Timesteps (Sensory Deprivation)",
       y = "Decoded State (Normalized)") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom",
        strip.background = element_rect(fill = "#f0f0f0", color = "black"),
        strip.text = element_text(face = "bold", size = 12),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
        panel.grid.minor = element_blank())

ggsave("results/Fig7_Thalamic_Feedback_Dynamics.png", plot = p, width = 12, height = 7, dpi = 300)
cat("Done! Plot saved to results/Fig7_Thalamic_Feedback_Dynamics.png\n")
