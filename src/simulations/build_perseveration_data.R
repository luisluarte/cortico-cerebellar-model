library(Rcpp)
library(dplyr)

cat("1. Building Enhanced Simulation Data...\n")
sourceCpp("src/r/sim_bio_probes.cpp")
targets <- readRDS("src/r/distillation_targets_v2.rds")
X_all <- targets$X
ITI_all <- targets$ITI / max(targets$ITI)
subjs <- targets$subjs
labels <- targets$labels
unique_subjs <- unique(subjs)

df_emp <- read.csv("data/behavioral_compilate.csv", stringsAsFactors = FALSE)
sel_p <- unique(df_emp$participant_id)[1:100]
df_sub <- df_emp[df_emp$participant_id %in% sel_p, ]
factor_ids <- levels(as.factor(df_sub$participant_id))

post_df <- readRDS("results/factorial_fixed_10k_A.rds")
params <- t(sapply(post_df[['bio_dist']][['traces']], function(x) exp(apply(x, 3, median))))

input_dim <- ncol(X_all)
set.seed(42)
W_gen <- matrix(rnorm(input_dim * 32, 0, 1/sqrt(32)), nrow=input_dim, ncol=32)
W_ach1 <- matrix(rnorm(2000 * 32, 0, 1/sqrt(32)), nrow=2000, ncol=32)
mask1 <- matrix(rbinom(2000 * 32, 1, 0.1), nrow=2000, ncol=32)
W_ach1 <- W_ach1 * mask1
W_ach2 <- matrix(rnorm(1000 * 2000, 0, 1/sqrt(2000)), nrow=1000, ncol=2000)
W_thal <- matrix(rnorm(32 * 1000, 0, 1/sqrt(1000)), nrow=32, ncol=1000)

long_df <- data.frame()
for(s_idx in 1:length(unique_subjs)) {
  s_num <- unique_subjs[s_idx]
  s_name <- factor_ids[s_num]
  idx_s <- which(subjs == s_num)
  X_s <- X_all[idx_s, , drop=FALSE]
  ITI_s <- ITI_all[idx_s]
  Pi_s <- matrix(1, nrow=length(idx_s), ncol=input_dim)
  choices_s <- labels[idx_s]
  
  d_emp <- df_emp[df_emp$participant_id == s_name, ] %>% filter(Resp %in% c(1, 2) & ((ttr - ttp) / 1000) > 0.1)
  if(nrow(d_emp) != length(idx_s)) {
      min_len <- min(nrow(d_emp), length(idx_s))
      d_emp <- d_emp[1:min_len, ]
      idx_s <- idx_s[1:min_len]
      X_s <- X_s[1:min_len, , drop=FALSE]
      ITI_s <- ITI_s[1:min_len]
      Pi_s <- Pi_s[1:min_len, , drop=FALSE]
      choices_s <- choices_s[1:min_len]
  }
  
  d_emp$diff_prob <- c(0, diff(d_emp$prob))
  rev_idx <- which(abs(d_emp$diff_prob) > 0.4)
  rev_window <- unique(unlist(lapply(rev_idx, function(x) seq(x, min(x+4, nrow(d_emp))))))
  
  opt_p <- params[s_idx, ]
  delta_beta <- opt_p[3]
  
  base_traces_opt <- simulate_bio_probes(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s,
                                         opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1)
  df_glm <- data.frame(Choice = choices_s, base_traces_opt$mu)
  suppressWarnings({ logit_model <- glm(Choice ~ ., data = df_glm, family = binomial(link="logit")) })
  p1_opt <- predict(logit_model, type="response")
  
  base_traces_les <- simulate_bio_probes(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s,
                                         opt_p[1], opt_p[2], 0.0, opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1)
  df_glm_les <- data.frame(Choice = choices_s, base_traces_les$mu)
  p1_les <- predict(logit_model, newdata = df_glm_les, type="response")
  
  t_all <- 2:nrow(d_emp)
  if(length(t_all) > 10) {
      prev_choice <- choices_s[t_all - 1]
      feedback_prev <- d_emp$F[t_all - 1]
      
      p_sw_opt <- ifelse(prev_choice == 1, 1 - p1_opt[t_all], p1_opt[t_all])
      p_sw_les <- ifelse(prev_choice == 1, 1 - p1_les[t_all], p1_les[t_all])
      
      state <- ifelse(t_all %in% rev_window, "Reversal", "Stable")
      
      df_opt <- data.frame(Subject=s_name, Trial=t_all, State=state, Condition="Optimized", 
                           P_Switch=p_sw_opt, Delta_Beta=delta_beta, Prev_Feedback=feedback_prev, Prev_Choice=prev_choice)
      df_les <- data.frame(Subject=s_name, Trial=t_all, State=state, Condition="Lesioned", 
                           P_Switch=p_sw_les, Delta_Beta=delta_beta, Prev_Feedback=feedback_prev, Prev_Choice=prev_choice)
      long_df <- rbind(long_df, df_opt, df_les)
  }
}
long_df <- long_df %>% filter(Delta_Beta <= 1.0)
long_df$P_Stay <- 1 - long_df$P_Switch

# Compute Loss Streak
long_df <- long_df %>% arrange(Subject, Condition, Trial)
long_df$Loss_Streak <- 0

cat("2. Computing Loss Streaks...\n")
subjects <- unique(long_df$Subject)
for(cond in c("Optimized", "Lesioned")) {
  for(subj in subjects) {
    idx <- which(long_df$Subject == subj & long_df$Condition == cond)
    if(length(idx) == 0) next
    
    streak <- 0
    for(i in seq_along(idx)) {
      curr_row <- idx[i]
      if(long_df$Prev_Feedback[curr_row] == 0) {
        if(i > 1 && long_df$Prev_Choice[curr_row] == long_df$Prev_Choice[idx[i-1]]) {
          streak <- streak + 1
        } else {
          streak <- 1
        }
      } else {
        streak <- 0
      }
      long_df$Loss_Streak[curr_row] <- streak
    }
  }
}

saveRDS(long_df, "results/perseveration_df.rds")
cat("Saved perseveration_df.rds\n")
