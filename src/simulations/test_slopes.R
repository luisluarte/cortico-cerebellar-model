library(Rcpp)
library(dplyr)
library(lme4)
library(lmerTest)
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

# We isolate the strict rising phase (T=0 to T=3) to test linear accumulation slopes
df_window <- df_loss %>% filter(Time_Since_Rev >= 0 & Time_Since_Rev <= 3)

df_long <- bind_rows(
  df_window %>% select(Subject, Time_Since_Rev, Delta_Beta, P_Switch = P_Opt) %>% mutate(Condition = "Optimized"),
  df_window %>% select(Subject, Time_Since_Rev, Delta_Beta, P_Switch = P_Les) %>% mutate(Condition = "Lesioned_Refit")
)
df_long$Condition <- factor(df_long$Condition, levels=c("Lesioned_Refit", "Optimized"))
df_long$Subject <- as.factor(df_long$Subject)

# Use Log-Odds to map perfectly to the Gaussian Logit GAMM framework
N <- nrow(df_long)
df_long$P_Switch <- (df_long$P_Switch * (N - 1) + 0.5) / N
df_long$Logit_P <- qlogis(df_long$P_Switch)

cat("\n============================================\n")
cat("TESTING LINEAR SLOPES DURING ADAPTATION PHASE (T=0 to T=3)\n")
cat("============================================\n")

# 1. Slope for Lesioned Network
fit_les <- lmer(Logit_P ~ Time_Since_Rev + (1 | Subject), data = df_long %>% filter(Condition == "Lesioned_Refit"))
cat("\n--- LESIONED NETWORK SLOPE ---\n")
print(summary(fit_les)$coefficients)

# 2. Slope for Intact Network
fit_opt <- lmer(Logit_P ~ Time_Since_Rev + (1 | Subject), data = df_long %>% filter(Condition == "Optimized"))
cat("\n--- INTACT NETWORK SLOPE ---\n")
print(summary(fit_opt)$coefficients)

# 3. Direct Interaction Test (Does the Intact slope significantly exceed the Lesion slope?)
fit_int <- lmer(Logit_P ~ Time_Since_Rev * Condition + (1 | Subject), data = df_long)
cat("\n--- INTERACTION (DIFFERENCE IN SLOPES) ---\n")
print(summary(fit_int)$coefficients)
