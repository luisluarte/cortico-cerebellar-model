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
df_window <- df_loss %>% filter(Time_Since_Rev >= 0 & Time_Since_Rev <= 15)

df_long <- bind_rows(
  df_window %>% select(Subject, Time_Since_Rev, Delta_Beta, P_Switch = P_Opt) %>% mutate(Condition = "Optimized"),
  df_window %>% select(Subject, Time_Since_Rev, Delta_Beta, P_Switch = P_Les) %>% mutate(Condition = "Lesioned_Refit")
)
df_long$Condition <- factor(df_long$Condition, levels=c("Optimized", "Lesioned_Refit"))
df_long$Subject <- as.factor(df_long$Subject)
N <- nrow(df_long)
df_long$P_Switch <- (df_long$P_Switch * (N - 1) + 0.5) / N

cat("Fitting GAMM...\n")
fit_cov <- gam(
  P_Switch ~ Condition + 
             s(Time_Since_Rev, by = Condition, k = 5) + 
             s(Delta_Beta, by = Condition, k = 5) + 
             s(Subject, bs = "re"),
  data = df_long,
  family = betar(link = "logit"),
  method = "REML"
)

cat("Extracting percentiles...\n")
db_75 <- quantile(df_long$Delta_Beta, 0.75)
db_50 <- quantile(df_long$Delta_Beta, 0.50)
db_25 <- quantile(df_long$Delta_Beta, 0.25)

get_preds <- function(target_db, label) {
  nd_opt <- data.frame(
    Time_Since_Rev = seq(0, 15, length.out = 100),
    Delta_Beta = target_db,
    Condition = factor("Optimized", levels = levels(df_long$Condition)),
    Subject = df_long$Subject[1]
  )
  nd_les <- nd_opt
  nd_les$Condition <- factor("Lesioned_Refit", levels = levels(df_long$Condition))
  
  preds_opt <- predict(fit_cov, newdata = nd_opt, type = "response", se.fit=TRUE, exclude="s(Subject)")
  preds_les <- predict(fit_cov, newdata = nd_les, type = "response", se.fit=TRUE, exclude="s(Subject)")
  
  df_curves <- data.frame(
    Time_Since_Rev = rep(nd_opt$Time_Since_Rev, 2),
    Condition = factor(rep(c("Optimized", "Lesioned_Refit"), each=100), levels=c("Optimized", "Lesioned_Refit")),
    Fit = c(preds_opt$fit, preds_les$fit),
    SE = c(preds_opt$se.fit, preds_les$se.fit),
    Percentile = label
  )
  df_curves$Lower <- df_curves$Fit - 1.96 * df_curves$SE
  df_curves$Upper <- df_curves$Fit + 1.96 * df_curves$SE
  return(df_curves)
}

df_75 <- get_preds(db_75, "75th Pctile (High Buffer)")
df_50 <- get_preds(db_50, "50th Pctile (Median Buffer)")
df_25 <- get_preds(db_25, "25th Pctile (Low Buffer)")

df_plot <- bind_rows(df_75, df_50, df_25)
df_plot$Percentile <- factor(df_plot$Percentile, levels = c(
  "25th Pctile (Low Buffer)", 
  "50th Pctile (Median Buffer)", 
  "75th Pctile (High Buffer)"
))

out_path <- "C:/Users/DCCS5/.gemini/antigravity/brain/ee7b6b70-a0ae-4607-9cdf-f55667b1cc2c/Fig80_GAMM_PostLoss_Panels.png"
p <- ggplot(df_plot, aes(x = Time_Since_Rev, y = Fit, color = Condition, fill = Condition)) +
  geom_line(linewidth = 1.2) +
  geom_ribbon(aes(ymin = Lower, ymax = Upper), alpha = 0.3, color = NA) +
  scale_color_manual(values = c("Lesioned_Refit" = "#e7298a", "Optimized" = "#1b9e77")) +
  scale_fill_manual(values = c("Lesioned_Refit" = "#e7298a", "Optimized" = "#1b9e77")) +
  facet_wrap(~Percentile) +
  theme_bw(base_size = 14) +
  labs(
    title = "Dose-Response: P(Switch | Loss)",
    subtitle = "Reversal Time-Course evaluated at different native Cerebellar phenotypes",
    x = "Trials Since Environmental Reversal",
    y = "P(Switch | Loss)"
  ) +
  theme(
    legend.position = "bottom",
    strip.background = element_rect(fill = "#f0f0f0", color = "black"),
    strip.text = element_text(face = "bold")
  )

ggsave(out_path, p, width=12, height=5, bg="white")
cat("Saved panel plots to", out_path, "\n")
