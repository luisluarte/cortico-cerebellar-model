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
cat("Processing...\n")
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
  p_les_acute <- suppressWarnings(predict(glm_opt, newdata = df_les, type="response"))
  
  glm_les <- suppressWarnings(glm(Choice ~ ., data = df_les, family = binomial(link="logit")))
  p_les_comp <- suppressWarnings(predict(glm_les, type="response"))
  
  df_subj <- data.frame(
    Subject = as.character(s_num),
    Feedback = ifelse(f_s == 1, "Win", "Loss"),
    P_Opt = p_opt,
    P_Les_Acute = p_les_acute,
    P_Les_Comp = p_les_comp
  )
  all_trials[[length(all_trials) + 1]] <- df_subj
}

df_all <- bind_rows(all_trials)

cat("\n--- Mean P(Switch | Feedback) ---\n")
summary_df <- df_all %>%
  group_by(Feedback) %>%
  summarize(
    Optimized = mean(P_Opt),
    Lesioned_Acute = mean(P_Les_Acute),
    Lesioned_Compensated = mean(P_Les_Comp)
  )
print(summary_df)

cat("\n--- P(Match WSLS) (Similarity to Rigid WSLS) ---\n")
wsls_df <- df_all %>%
  mutate(
    WSLS_Opt = ifelse(Feedback == "Win", 1 - P_Opt, P_Opt),
    WSLS_Acute = ifelse(Feedback == "Win", 1 - P_Les_Acute, P_Les_Acute),
    WSLS_Comp = ifelse(Feedback == "Win", 1 - P_Les_Comp, P_Les_Comp)
  ) %>%
  summarize(
    Optimized = mean(WSLS_Opt),
    Lesioned_Acute = mean(WSLS_Acute),
    Lesioned_Compensated = mean(WSLS_Comp)
  )
print(wsls_df)
