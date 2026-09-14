library(Rcpp)
library(dplyr)
library(mgcv)
library(ggplot2)
sourceCpp("src/r/sim_bio_probes.cpp")

cat("Preparing Data (Top 50% Native Beta)...\n")
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
  
  s_emp <- d_emp[d_emp$participant_id == s_name, ]
  s_emp$prob_diff <- c(0, diff(s_emp$prob))
  rev_trials <- which(abs(s_emp$prob_diff) > 0.4)
  valid_emp_idx <- which(s_emp$Resp %in% c(1, 2) & ((s_emp$ttr - s_emp$ttp) / 1000) > 0.1)
  
  time_since_rev <- rep(NA, length(idx_s))
  for(i in 1:length(idx_s)) {
    t <- valid_emp_idx[i]
    past_revs <- rev_trials[rev_trials <= t]
    if(length(past_revs) > 0) time_since_rev[i] <- t - max(past_revs)
  }
  all_trials[[length(all_trials) + 1]] <- data.frame(Subject = s_name, Time_Since_Rev = time_since_rev, Delta_Beta = opt_p[3], P_Opt = p_opt, P_Les = p_les_refit)
}

df_all <- bind_rows(all_trials)
med_db <- median(unique(df_all$Delta_Beta))

# IMPORTANT: > med_db (Isolating the 25 participants who rely on the cerebellum)
df_window <- df_all %>% filter(Time_Since_Rev >= 0 & Time_Since_Rev <= 15 & Delta_Beta > med_db)

df_long <- bind_rows(
  df_window %>% select(Subject, Time_Since_Rev, P_Switch = P_Opt) %>% mutate(Condition = "Optimized"),
  df_window %>% select(Subject, Time_Since_Rev, P_Switch = P_Les) %>% mutate(Condition = "Lesioned_Refit")
)
df_long$Condition <- factor(df_long$Condition, levels=c("Optimized", "Lesioned_Refit"))
df_long$Subject <- as.factor(df_long$Subject)
N <- nrow(df_long)
df_long$P_Switch <- (df_long$P_Switch * (N - 1) + 0.5) / N

cat("Fitting Option 1 GAMM...\n")
fit_opt1 <- gam(
  P_Switch ~ Condition + 
             s(Time_Since_Rev, by = Condition, k = 5) + 
             s(Subject, bs = "re"),
  data = df_long,
  family = betar(link = "logit"),
  method = "REML"
)

nd_opt <- data.frame(
  Time_Since_Rev = seq(0, 15, length.out = 100),
  Condition = factor("Optimized", levels = levels(df_long$Condition)),
  Subject = df_long$Subject[1] # Dummy
)
nd_les <- nd_opt
nd_les$Condition <- factor("Lesioned_Refit", levels = levels(df_long$Condition))

Xp_opt <- predict(fit_opt1, newdata = nd_opt, type = "lpmatrix", exclude = "s(Subject)")
Xp_les <- predict(fit_opt1, newdata = nd_les, type = "lpmatrix", exclude = "s(Subject)")

Xp_diff <- Xp_opt - Xp_les
diff_est <- Xp_diff %*% coef(fit_opt1)
diff_se <- sqrt(rowSums((Xp_diff %*% vcov(fit_opt1)) * Xp_diff))

df_plot <- data.frame(
  Time_Since_Rev = nd_opt$Time_Since_Rev,
  Diff = as.numeric(diff_est),
  CI_lower = as.numeric(diff_est - 1.96 * diff_se),
  CI_upper = as.numeric(diff_est + 1.96 * diff_se)
)

out_path <- "C:/Users/DCCS5/.gemini/antigravity/brain/ee7b6b70-a0ae-4607-9cdf-f55667b1cc2c/Fig75_GAMM_Option1_Difference.png"

p <- ggplot(df_plot, aes(x = Time_Since_Rev, y = Diff)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 1) +
  geom_ribbon(aes(ymin = CI_lower, ymax = CI_upper), fill = "#E7298A", alpha = 0.3) +
  geom_line(color = "#E7298A", linewidth = 1.5) +
  theme_minimal(base_size = 14) +
  labs(
    title = "Contrast Curve: High Native Cerebellar Buffer (N=25)",
    subtitle = "Difference in Log-Odds (Intact - Lesioned)",
    x = "Trials Since Environmental Reversal",
    y = "\u0394 Log-Odds (Intact - Lesion)"
  ) +
  annotate("text", x = 10, y = 0.1, label = "0 = No Difference", color = "black", fontface = "bold")

ggsave(out_path, p, width = 8, height = 5, bg = "white")
cat("Plot saved to", out_path, "\n")
