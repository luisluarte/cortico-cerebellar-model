library(Rcpp)
library(RcppArmadillo)
library(ggplot2)
library(dplyr)
library(tidyr)
library(ggpubr)

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

res_df <- data.frame()

cat("Running State-Dependent Synthetic Lesions...\n")
for(s_idx in 1:length(unique_subjs)) {
  s_num <- unique_subjs[s_idx]
  s_name <- factor_ids[s_num] # MAP NUMERIC BACK TO STRING
  
  idx_s <- which(subjs == s_num)
  X_s <- X_all[idx_s, , drop=FALSE]
  ITI_s <- ITI_all[idx_s]
  Pi_s <- matrix(1, nrow=length(idx_s), ncol=input_dim)
  choices_s <- labels[idx_s]
  
  d_emp <- df_emp[df_emp$participant_id == s_name, ]
  
  # Wait! The human trials in d_emp might include skipped/invalid trials. 
  # In generate_stan_data.R they did: filter(Resp %in% c(1, 2) & rt > 0.1)
  d_emp <- d_emp %>% filter(Resp %in% c(1, 2) & ((ttr - ttp) / 1000) > 0.1)
  
  # Make sure dimensions match!
  if(nrow(d_emp) != length(idx_s)) {
      cat("Warning: Mismatch for", s_name, "- Emp:", nrow(d_emp), "Tgt:", length(idx_s), "\n")
      # Truncate to min
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
  stable_window <- setdiff(1:nrow(d_emp), rev_window)
  
  opt_p <- params[s_idx, ]
  
  # 1. OPTIMIZED STATE
  base_traces_opt <- simulate_bio_probes(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s, 
                                     opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1)
  
  df_glm <- data.frame(Choice = choices_s, base_traces_opt$mu)
  suppressWarnings({
    logit_model <- glm(Choice ~ ., data = df_glm, family = binomial(link="logit"))
  })
  
  p_choose1_opt <- predict(logit_model, type="response")
  prev_choice <- c(NA, choices_s[1:(length(choices_s)-1)])
  p_pers_opt <- ifelse(prev_choice == 1, p_choose1_opt, 1 - p_choose1_opt)
  
  # 2. LESIONED STATE (beta_thal = 0)
  base_traces_les <- simulate_bio_probes(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s, 
                                     opt_p[1], opt_p[2], 0.0, opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1)
  
  df_glm_les <- data.frame(Choice = choices_s, base_traces_les$mu)
  p_choose1_les <- predict(logit_model, newdata = df_glm_les, type="response")
  p_pers_les <- ifelse(prev_choice == 1, p_choose1_les, 1 - p_choose1_les)
  
  # 3. COMPUTE AVERAGES FOR ERROR TRIALS
  prev_reward <- c(NA, d_emp$F[1:(nrow(d_emp)-1)])
  error_stable <- intersect(stable_window, which(prev_reward == 0))
  error_rev <- intersect(rev_window, which(prev_reward == 0))
  
  res_df <- rbind(res_df, data.frame(
    Subject = s_name,
    State = "Stable (Markovian)",
    Condition = "Optimized",
    Perseveration = mean(p_pers_opt[error_stable], na.rm=TRUE)
  ))
  res_df <- rbind(res_df, data.frame(
    Subject = s_name,
    State = "Stable (Markovian)",
    Condition = "Lesioned",
    Perseveration = mean(p_pers_les[error_stable], na.rm=TRUE)
  ))
  
  res_df <- rbind(res_df, data.frame(
    Subject = s_name,
    State = "Reversal (Non-Markovian)",
    Condition = "Optimized",
    Perseveration = mean(p_pers_opt[error_rev], na.rm=TRUE)
  ))
  res_df <- rbind(res_df, data.frame(
    Subject = s_name,
    State = "Reversal (Non-Markovian)",
    Condition = "Lesioned",
    Perseveration = mean(p_pers_les[error_rev], na.rm=TRUE)
  ))
}

cat("Generating Plot...\n")
res_df <- na.omit(res_df) # Just in case some had 0 reversal errors
res_df$Condition <- factor(res_df$Condition, levels = c("Optimized", "Lesioned"))
res_df$State <- factor(res_df$State, levels = c("Stable (Markovian)", "Reversal (Non-Markovian)"))

p <- ggplot(res_df, aes(x = Condition, y = Perseveration, fill = Condition)) +
  geom_boxplot(alpha = 0.5, outlier.shape = NA, width = 0.5) +
  geom_point(aes(color = Condition), position = position_jitter(width=0.05), alpha = 0.7) +
  geom_line(aes(group = Subject), alpha = 0.2, color = "gray50") +
  facet_wrap(~ State) +
  scale_color_manual(values = c("Optimized" = "#d95f02", "Lesioned" = "#1b9e77")) +
  scale_fill_manual(values = c("Optimized" = "#d95f02", "Lesioned" = "#1b9e77")) +
  theme_minimal(base_size = 14) +
  labs(
    title = "Causal Lesion of Thalamic Rigidity",
    subtitle = "Ablating beta_thal destroys Stable perseveration, rather than dynamically adapting it",
    x = "Network Connectivity",
    y = "P(Stay | Previous Loss)"
  ) +
  theme(legend.position = "none", panel.spacing = unit(2, "lines"))

ggsave("results/Fig20_Lesion_States.png", plot = p, width = 8, height = 5, dpi = 300)
cat("Done. Saved to results/Fig20_Lesion_States.png\n")

# Tests
w_stable <- wilcox.test(
  res_df$Perseveration[res_df$Condition == "Optimized" & res_df$State == "Stable (Markovian)"], 
  res_df$Perseveration[res_df$Condition == "Lesioned" & res_df$State == "Stable (Markovian)"], 
  paired = TRUE
)
w_rev <- wilcox.test(
  res_df$Perseveration[res_df$Condition == "Optimized" & res_df$State == "Reversal (Non-Markovian)"], 
  res_df$Perseveration[res_df$Condition == "Lesioned" & res_df$State == "Reversal (Non-Markovian)"], 
  paired = TRUE
)

cat("\n--- Stable (Markovian) ---\n")
print(w_stable)
cat("\n--- Reversal (Non-Markovian) ---\n")
print(w_rev)
