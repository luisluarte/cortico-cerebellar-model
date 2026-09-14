library(Rcpp)
library(RcppArmadillo)
library(dplyr)

cat("1. Building Continuous Simulation Data (All Trials)...\n")
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

long_df_all <- data.frame()
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
      # Using the continuous sequence of P(Switch) regardless of whether they just won or lost
      p_sw_opt <- ifelse(prev_choice == 1, 1 - p1_opt[t_all], p1_opt[t_all])
      p_sw_les <- ifelse(prev_choice == 1, 1 - p1_les[t_all], p1_les[t_all])
      
      df_opt <- data.frame(Subject=s_name, Trial=t_all, Condition="Optimized", P_Switch=p_sw_opt, Delta_Beta=delta_beta)
      df_les <- data.frame(Subject=s_name, Trial=t_all, Condition="Lesioned", P_Switch=p_sw_les, Delta_Beta=delta_beta)
      long_df_all <- rbind(long_df_all, df_opt, df_les)
  }
}
long_df_all <- long_df_all %>% filter(Delta_Beta <= 1.0)
saveRDS(long_df_all, "results/long_df_all.rds")
cat("Saved long_df_all.rds\n")
