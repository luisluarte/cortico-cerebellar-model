library(Rcpp)
library(dplyr)
library(mgcv)
library(ggplot2)
sourceCpp("src/r/sim_bio_probes.cpp")

cat("Preparing Data...\n")
targets <- readRDS("src/r/distillation_targets_v2.rds")
d_emp <- read.csv("data/behavioral_compilate.csv", stringsAsFactors = FALSE)
post_df <- readRDS("results/factorial_fixed_10k_A.rds")
params <- t(sapply(post_df[['bio_dist']][['traces']], function(x) exp(apply(x, 3, median))))
input_dim <- ncol(targets$X)

set.seed(42)
W_gen <- matrix(rnorm(input_dim * 32, 0, 1/sqrt(32)), nrow=input_dim, ncol=32)
W_ach1 <- matrix(rnorm(2000 * 32, 0, 1/sqrt(32)), nrow=2000, ncol=32)
mask1 <- matrix(rbinom(2000 * 32, 1, 0.1), nrow=2000, ncol=32)
W_ach1 <- W_ach1 * mask1
W_ach2 <- matrix(rnorm(1000 * 2000, 0, 1/sqrt(2000)), nrow=1000, ncol=2000)
W_thal <- matrix(rnorm(32 * 1000, 0, 1/sqrt(1000)), nrow=32, ncol=1000)

unique_subjs <- unique(targets$subjs)
factor_ids <- levels(as.factor(d_emp$participant_id[d_emp$participant_id %in% unique(d_emp$participant_id)[1:100]]))

all_trials <- list()
for(s_idx in 1:length(unique_subjs)) {
  s_num <- unique_subjs[s_idx]
  s_name <- factor_ids[s_num]
  idx_s <- which(targets$subjs == s_num)
  if(length(idx_s) < 50) next
  X_s <- matrix(as.numeric(targets$X[idx_s, ]), ncol=input_dim)
  ITI_s <- targets$ITI[idx_s] / max(targets$ITI)
  Pi_s <- matrix(1, nrow=length(idx_s), ncol=input_dim)
  choices_s <- targets$labels[idx_s]
  opt_p <- params[s_idx, ]
  
  tr_opt <- simulate_bio_probes(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s,
                                opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1)
  df_opt <- data.frame(Choice = choices_s, tr_opt$mu)
  glm_opt <- suppressWarnings(glm(Choice ~ ., data = df_opt, family = binomial(link="logit")))
  p_opt <- suppressWarnings(predict(glm_opt, type="response"))
  
  tr_les <- simulate_bio_probes(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s,
                                opt_p[1], opt_p[2], 0.0, opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1)
  df_les <- data.frame(Choice = choices_s, tr_les$mu)
  glm_les <- suppressWarnings(glm(Choice ~ ., data = df_les, family = binomial(link="logit")))
  p_les_refit <- suppressWarnings(predict(glm_les, type="response"))
  
  prev_choice <- c(NA, choices_s[-length(choices_s)])
  true_p_switch_opt <- ifelse(prev_choice == 1, 1 - p_opt, p_opt)
  true_p_switch_les <- ifelse(prev_choice == 1, 1 - p_les_refit, p_les_refit)
  
  s_emp <- d_emp[d_emp$participant_id == s_name, ]
  valid_emp_idx <- which(s_emp$Resp %in% c(1, 2) & ((s_emp$ttr - s_emp$ttp) / 1000) > 0.1)
  
  F_seq <- s_emp$F[valid_emp_idx]
  if(length(F_seq) > length(idx_s)) F_seq <- F_seq[1:length(idx_s)]
  prev_F <- c(NA, F_seq[-length(F_seq)])
  if(length(prev_F) < length(idx_s)) prev_F <- c(prev_F, rep(NA, length(idx_s) - length(prev_F)))
  
  s_emp$prob_diff <- c(0, diff(s_emp$prob))
  rev_trials <- which(abs(s_emp$prob_diff) > 0.4)
  time_since_rev <- rep(NA, length(idx_s))
  for(i in 1:length(idx_s)) {
    t <- valid_emp_idx[i]
    past_revs <- rev_trials[rev_trials <= t]
    if(length(past_revs) > 0) time_since_rev[i] <- t - max(past_revs)
  }
  
  all_trials[[length(all_trials) + 1]] <- data.frame(
    Subject = s_name, 
    Time_Since_Rev = time_since_rev, 
    Delta_Beta = opt_p[3], 
    Prev_F = prev_F,
    P_Opt = true_p_switch_opt, 
    P_Les = true_p_switch_les
  )
}

df_all <- bind_rows(all_trials)
df_loss <- df_all %>% filter(!is.na(Prev_F) & Prev_F == 0)
df_window <- df_loss %>% filter(Time_Since_Rev >= 0 & Time_Since_Rev <= 5)

df_long <- bind_rows(
  df_window %>% select(Subject, Time_Since_Rev, Delta_Beta, P_Switch = P_Opt) %>% mutate(Condition = "Optimized"),
  df_window %>% select(Subject, Time_Since_Rev, Delta_Beta, P_Switch = P_Les) %>% mutate(Condition = "Lesioned_Refit")
)
df_long$Condition <- factor(df_long$Condition, levels=c("Optimized", "Lesioned_Refit"))
df_long$Subject <- as.factor(df_long$Subject)
N <- nrow(df_long)
df_long$P_Switch <- (df_long$P_Switch * (N - 1) + 0.5) / N
df_long$Logit_P <- qlogis(df_long$P_Switch)

cat("Fitting Optimized Gaussian GAMM...\n")
fit_cov <- gam(
  Logit_P ~ Condition + 
             s(Time_Since_Rev, by = Condition, k = 5) + 
             s(Delta_Beta, by = Condition, k = 5) + 
             s(Subject, bs = "re"),
  data = df_long,
  family = gaussian(),
  method = "REML"
)

cat("Extracting p-values across percentiles...\n")
pcts <- seq(0.10, 0.90, by = 0.01)
res_list <- list()

for(p in pcts) {
  target_db <- quantile(df_long$Delta_Beta, p)
  
  nd_opt <- data.frame(Time_Since_Rev = 3, Delta_Beta = target_db, Condition = factor("Optimized", levels = levels(df_long$Condition)), Subject = df_long$Subject[1])
  nd_les <- data.frame(Time_Since_Rev = 3, Delta_Beta = target_db, Condition = factor("Lesioned_Refit", levels = levels(df_long$Condition)), Subject = df_long$Subject[1])
  
  Xp_opt <- predict(fit_cov, newdata = nd_opt, type = "lpmatrix", exclude = "s(Subject)")
  Xp_les <- predict(fit_cov, newdata = nd_les, type = "lpmatrix", exclude = "s(Subject)")
  
  Xp_diff <- Xp_opt - Xp_les
  diff_est <- as.numeric(Xp_diff %*% coef(fit_cov))
  diff_se <- sqrt(rowSums((Xp_diff %*% vcov(fit_cov)) * Xp_diff))
  
  z_score <- diff_est / diff_se
  p_val <- 2 * pnorm(-abs(z_score))
  
  res_list[[length(res_list)+1]] <- data.frame(Percentile = p * 100, P_Value = p_val)
}
df_p <- bind_rows(res_list)

out_path <- "C:/Users/DCCS5/.gemini/antigravity/brain/ee7b6b70-a0ae-4607-9cdf-f55667b1cc2c/Fig85_PValue_Valley.png"

p_plot <- ggplot(df_p, aes(x = Percentile, y = P_Value)) +
  # Fill the significant area
  geom_ribbon(data = df_p %>% filter(P_Value <= 0.05), aes(ymin = 0, ymax = P_Value), fill = "red", alpha = 0.2) +
  # Threshold lines
  geom_hline(yintercept = 0.05, linetype = "dashed", color = "red", linewidth = 1) +
  geom_hline(yintercept = 0.10, linetype = "dotted", color = "gray50", linewidth = 1) +
  # Main curve
  geom_line(linewidth = 2, color = "#2c3e50") +
  scale_x_continuous(breaks = seq(10, 90, by=10)) +
  scale_y_continuous(breaks = c(0, 0.05, 0.10, 0.20, 0.30, 0.40)) +
  expand_limits(y = 0) +
  theme_minimal(base_size = 14) +
  labs(
    title = "The 'Significance Valley' at Trial 3",
    subtitle = "Pointwise p-values testing Lesion effect across the entire population distribution",
    x = "Population Percentile (\u0394\u03B2)",
    y = "P-Value (Intact vs. Lesioned)"
  ) +
  annotate("text", x = 50, y = 0.02, label = "p < 0.05\n(Significant)", color = "red", fontface = "bold", size = 5)

ggsave(out_path, p_plot, width=9, height=5, bg="white")
cat("Saved Significance Valley Plot.\n")
