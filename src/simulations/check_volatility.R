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

all_trials <- list()
for(s_idx in 1:length(unique(targets$subjs))) {
  s_num <- unique(targets$subjs)[s_idx]
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
  p_les_comp <- suppressWarnings(predict(glm_les, type="response"))
  
  prev_choices <- c(NA, choices_s[1:(length(choices_s)-1)])
  p_switch_opt <- ifelse(prev_choices == 1, 1 - p_opt, p_opt)
  p_switch_les <- ifelse(prev_choices == 1, 1 - p_les_comp, p_les_comp)

  df_subj <- data.frame(
    Subject = as.character(s_num),
    P_Opt = p_switch_opt, P_Les_Comp = p_switch_les
  )
  # Calculate trial-to-trial volatility (absolute derivative)
  df_subj$Delta_Opt <- c(NA, abs(diff(df_subj$P_Opt)))
  df_subj$Delta_Les <- c(NA, abs(diff(df_subj$P_Les_Comp)))
  
  all_trials[[length(all_trials) + 1]] <- df_subj
}

df_all <- bind_rows(all_trials) %>% filter(!is.na(Delta_Opt))

cat("\n--- Trial-to-Trial Volatility (Rigidity vs Integration) ---\n")
print(df_all %>% summarize(
  Mean_Jump_Intact = mean(Delta_Opt),
  Mean_Jump_Lesioned = mean(Delta_Les),
  Max_Jump_Intact = max(Delta_Opt),
  Max_Jump_Lesioned = max(Delta_Les)
))

# Formal paired t-test
t_res <- t.test(df_all$Delta_Les, df_all$Delta_Opt, paired=TRUE)
cat(sprintf("\nT-Test p-value: %e\n", t_res$p.value))
