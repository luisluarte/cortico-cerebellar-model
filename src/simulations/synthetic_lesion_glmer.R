library(Rcpp)
library(RcppArmadillo)
library(dplyr)
library(tidyr)
library(glmmTMB)

sourceCpp("src/r/sim_bio_probes.cpp")

cat("Loading Targets and Posteriors...\n")
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
colnames(params) <- c('alpha_pc', 'lambda_pc', 'beta_thal', 'kappa_cf', 'alpha_gran', 'beta_gran', 'sigma2')

set.seed(42)
input_dim <- ncol(X_all)
W_gen <- matrix(rnorm(input_dim * 32, 0, 1/sqrt(32)), nrow=input_dim, ncol=32)
W_ach1 <- matrix(rnorm(2000 * 32, 0, 1/sqrt(32)), nrow=2000, ncol=32)
mask1 <- matrix(rbinom(2000 * 32, 1, 0.1), nrow=2000, ncol=32)
W_ach1 <- W_ach1 * mask1
W_ach2 <- matrix(rnorm(1000 * 2000, 0, 1/sqrt(2000)), nrow=1000, ncol=2000)
W_thal <- matrix(rnorm(32 * 1000, 0, 1/sqrt(1000)), nrow=32, ncol=1000)

long_df <- data.frame()

cat("Extracting t+1 P(Switch) probabilities...\n")
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
  
  # State definitions
  d_emp$diff_prob <- c(0, diff(d_emp$prob))
  rev_idx <- which(abs(d_emp$diff_prob) > 0.4)
  rev_window <- unique(unlist(lapply(rev_idx, function(x) seq(x, min(x+4, nrow(d_emp))))))
  
  opt_p <- params[s_idx, ]
  delta_beta <- opt_p[3]
  
  # Opt Probabilities
  base_traces_opt <- simulate_bio_probes(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s, 
                                     opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1)
  df_glm <- data.frame(Choice = choices_s, base_traces_opt$mu)
  suppressWarnings({ logit_model <- glm(Choice ~ ., data = df_glm, family = binomial(link="logit")) })
  p1_opt <- predict(logit_model, type="response")
  
  # Les Probabilities
  base_traces_les <- simulate_bio_probes(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s, 
                                     opt_p[1], opt_p[2], 0.0, opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1)
  df_glm_les <- data.frame(Choice = choices_s, base_traces_les$mu)
  p1_les <- predict(logit_model, newdata = df_glm_les, type="response")
  
  # Find t+1 trials following a loss
  is_loss <- (d_emp$F == 0)
  t_plus_1 <- which(is_loss) + 1
  t_plus_1 <- t_plus_1[t_plus_1 <= nrow(d_emp)] # remove if loss was on final trial
  
  if(length(t_plus_1) > 0) {
      prev_choice <- choices_s[t_plus_1 - 1]
      
      # Probability of switching from the choice at t
      p_switch_opt <- ifelse(prev_choice == 1, 1 - p1_opt[t_plus_1], p1_opt[t_plus_1])
      p_switch_les <- ifelse(prev_choice == 1, 1 - p1_les[t_plus_1], p1_les[t_plus_1])
      
      state <- ifelse(t_plus_1 %in% rev_window, "Reversal", "Stable")
      
      df_opt <- data.frame(Subject=s_name, Trial=t_plus_1, State=state, 
                           Condition="Optimized", P_Switch=p_switch_opt, Delta_Beta=delta_beta)
      df_les <- data.frame(Subject=s_name, Trial=t_plus_1, State=state, 
                           Condition="Lesioned", P_Switch=p_switch_les, Delta_Beta=delta_beta)
                           
      long_df <- rbind(long_df, df_opt, df_les)
  }
}

long_df <- na.omit(long_df)
long_df$Condition <- factor(long_df$Condition, levels=c("Lesioned", "Optimized"))
long_df$State <- factor(long_df$State, levels=c("Stable", "Reversal"))

# Beta regression requires values strictly in (0,1)
long_df$P_Switch <- pmax(pmin(long_df$P_Switch, 0.9999), 0.0001)

cat("Fitting Continuous Beta Mixed Model...\n")
# Using Beta family to strictly model the continuous probabilities bounded between 0 and 1
m <- glmmTMB(P_Switch ~ Condition * State * Delta_Beta + (1 | Subject), 
             data = long_df, family = beta_family(link="logit"))

sink("results/glmer_switch_summary.txt")
print(summary(m))
sink()

cat("Done. Summary saved to results/glmer_switch_summary.txt\n")

library(emmeans)
cat('Computing emmeans...\n')
sink('results/glmer_switch_emmeans.txt')
emm <- emmeans(m, ~ Condition | State)
cat('=== Marginal Means (Logit Scale) ===\n')
print(emm)
cat('\n=== Marginal Means (Response Probability Scale) ===\n')
print(emmeans(m, ~ Condition | State, type = 'response'))
cat('\n=== Pairwise Contrasts ===\n')
print(pairs(emm))
sink()


cat("\n=== Pairwise Contrasts at specific Delta_Beta values ===\n")
db_seq <- seq(min(long_df$Delta_Beta), max(long_df$Delta_Beta), length.out = 8)
emm_grid <- emmeans(m, ~ Condition | State * Delta_Beta, at = list(Delta_Beta = db_seq))
sink("results/glmer_switch_emmeans_grid.txt")
print(pairs(emm_grid))
sink()


cat("\n=== Pairwise Contrasts at Quartiles of Delta_Beta ===\n")
db_seq <- unname(quantile(long_df$Delta_Beta, probs = c(0.25, 0.5, 0.75)))
cat("Q1, Q2, Q3:", db_seq, "\n")
emm_grid <- emmeans(m, ~ Condition | State * Delta_Beta, at = list(Delta_Beta = db_seq))
sink("results/glmer_switch_emmeans_quartiles.txt")
print(pairs(emm_grid))
sink()


cat("\n=== ROBUST ITERATION: Log-Transforming Delta_Beta ===\n")
long_df$log_Delta_Beta <- log(long_df$Delta_Beta)
m_robust <- glmmTMB(P_Switch ~ Condition * State * log_Delta_Beta + (1 | Subject), 
             data = long_df, family = beta_family(link="logit"))

sink("results/glmer_switch_summary_robust.txt")
print(summary(m_robust))
sink()

emm_robust <- emmeans(m_robust, ~ Condition | State)
sink("results/glmer_switch_emmeans_robust.txt")
print(pairs(emm_robust))
sink()


cat("\n=== Robust Pairwise Contrasts at Quartiles of log(Delta_Beta) ===\n")
log_db_seq <- unname(quantile(long_df$log_Delta_Beta, probs = c(0.25, 0.5, 0.75)))
cat("Q1, Q2, Q3 (log scale):", log_db_seq, "\n")
emm_robust_q <- emmeans(m_robust, ~ Condition | State * log_Delta_Beta, at = list(log_Delta_Beta = log_db_seq))
sink("results/glmer_switch_emmeans_robust_quartiles.txt")
print(pairs(emm_robust_q))
sink()


cat("\n=== DENSITY-WEIGHTED ITERATION ===\n")
dens <- density(long_df$Delta_Beta)
long_df$density_weight <- approx(dens$x, dens$y, xout = long_df$Delta_Beta)$y
# Normalize weights so the sum equals the original N to preserve standard errors properly
long_df$density_weight <- long_df$density_weight / mean(long_df$density_weight)

m_weighted <- glmmTMB(P_Switch ~ Condition * State * Delta_Beta + (1 | Subject), 
             data = long_df, family = beta_family(link="logit"),
             weights = density_weight)

sink("results/glmer_switch_summary_weighted.txt")
print(summary(m_weighted))
sink()

emm_weighted <- emmeans(m_weighted, ~ Condition | State)
sink("results/glmer_switch_emmeans_weighted.txt")
print(pairs(emm_weighted))
sink()

