library(Rcpp)
library(dplyr)
sourceCpp("src/r/sim_bio_probes.cpp")

targets <- readRDS("src/r/distillation_targets_v2.rds")
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
all_trials <- list()

for(s_idx in 1:length(unique_subjs)) {
  s_num <- unique_subjs[s_idx]
  idx_s <- which(targets$subjs == s_num)
  if(length(idx_s) < 50) next
  
  X_s <- matrix(as.numeric(targets$X[idx_s, ]), ncol=input_dim)
  ITI_s <- targets$ITI[idx_s] / max(targets$ITI)
  Pi_s <- matrix(1, nrow=length(idx_s), ncol=input_dim)
  choices_s <- targets$labels[idx_s]
  f_s <- targets$lag_reward[idx_s]
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
  p_les_comp <- suppressWarnings(predict(glm_les, type="response"))
  
  # Calculate Streak Lengths
  loss_streak <- rep(0, length(f_s))
  current_streak <- 0
  for(i in 1:length(f_s)) {
    if(f_s[i] == 0) {
      current_streak <- current_streak + 1
    } else {
      current_streak <- 0
    }
    loss_streak[i] <- current_streak
  }
  
  # For probability of switching, we need it relative to the PREVIOUS choice
  # Just like we did earlier: P(Switch) = 1 - p(Stay)
  # BUT since glm predicts Choice (0 or 1), we need to know what choice they made.
  # If Choice(t-1) == 1, then P(Switch) = P(Choice(t) == 0) = 1 - p_opt
  prev_choices <- c(NA, choices_s[1:(length(choices_s)-1)])
  p_switch_opt <- ifelse(prev_choices == 1, 1 - p_opt, p_opt)
  p_switch_les <- ifelse(prev_choices == 1, 1 - p_les_comp, p_les_comp)

  df_subj <- data.frame(
    Subject = as.character(s_num),
    Feedback = ifelse(f_s == 1, "Win", "Loss"),
    Loss_Streak = loss_streak,
    P_Opt = p_switch_opt,
    P_Les_Comp = p_switch_les
  )
  all_trials[[length(all_trials) + 1]] <- df_subj
}

df_all <- bind_rows(all_trials) %>% filter(!is.na(P_Opt))

df_losses <- df_all %>% filter(Feedback == "Loss")
df_losses$Streak_Bin <- ifelse(df_losses$Loss_Streak >= 3, "3+", as.character(df_losses$Loss_Streak))

cat("\n--- P(Switch) by Loss Streak Length ---\n")
summary_df <- df_losses %>%
  group_by(Streak_Bin) %>%
  summarize(
    N_Trials = n(),
    Intact_Dynamic = mean(P_Opt, na.rm=TRUE),
    Lesioned_Rigid = mean(P_Les_Comp, na.rm=TRUE)
  )
print(summary_df)
