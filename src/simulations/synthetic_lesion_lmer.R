library(Rcpp)
library(RcppArmadillo)
library(ggplot2)
library(dplyr)
library(tidyr)

# Ensure lme4 and lmerTest are available
if(!require(lme4)) install.packages("lme4", repos="https://cloud.r-project.org")
if(!require(lmerTest)) install.packages("lmerTest", repos="https://cloud.r-project.org")
library(lme4)
library(lmerTest)

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

trial_df <- data.frame()

cat("Extracting Trial-by-Trial Probabilities...\n")
for(s_idx in 1:length(unique_subjs)) {
  s_num <- unique_subjs[s_idx]
  s_name <- factor_ids[s_num]
  
  idx_s <- which(subjs == s_num)
  X_s <- X_all[idx_s, , drop=FALSE]
  ITI_s <- ITI_all[idx_s]
  Pi_s <- matrix(1, nrow=length(idx_s), ncol=input_dim)
  choices_s <- labels[idx_s]
  
  d_emp <- df_emp[df_emp$participant_id == s_name, ]
  d_emp <- d_emp %>% filter(Resp %in% c(1, 2) & ((ttr - ttp) / 1000) > 0.1)
  
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
  delta_beta <- opt_p[3] # Original beta_thal magnitude
  
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
  
  # 2. LESIONED STATE
  base_traces_les <- simulate_bio_probes(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s, 
                                     opt_p[1], opt_p[2], 0.0, opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1)
  
  df_glm_les <- data.frame(Choice = choices_s, base_traces_les$mu)
  p_choose1_les <- predict(logit_model, newdata = df_glm_les, type="response")
  p_pers_les <- ifelse(prev_choice == 1, p_choose1_les, 1 - p_choose1_les)
  
  # 3. FILTER FOR LOSS TRIALS
  prev_reward <- c(NA, d_emp$F[1:(nrow(d_emp)-1)])
  loss_idx <- which(prev_reward == 0)
  
  if(length(loss_idx) > 0) {
    df_subj_opt <- data.frame(
      Subject = s_name,
      Trial = loss_idx,
      Condition = "Optimized",
      State = ifelse(loss_idx %in% rev_window, "Reversal", "Stable"),
      P_Stay = p_pers_opt[loss_idx],
      Delta_Beta = delta_beta
    )
    
    df_subj_les <- data.frame(
      Subject = s_name,
      Trial = loss_idx,
      Condition = "Lesioned",
      State = ifelse(loss_idx %in% rev_window, "Reversal", "Stable"),
      P_Stay = p_pers_les[loss_idx],
      Delta_Beta = delta_beta
    )
    
    trial_df <- rbind(trial_df, df_subj_opt, df_subj_les)
  }
}

trial_df <- na.omit(trial_df)
trial_df$Condition <- factor(trial_df$Condition, levels = c("Lesioned", "Optimized"))
trial_df$State <- factor(trial_df$State, levels = c("Stable", "Reversal"))

cat("Running Mixed-Effects Model...\n")
# We use logit-transformed probabilities for LMM to satisfy normality, bounded [0.001, 0.999]
trial_df$logit_P <- qlogis(pmin(pmax(trial_df$P_Stay, 0.001), 0.999))

m <- lmer(logit_P ~ Condition * State * Delta_Beta + (1 | Subject), data = trial_df)
sink("results/lmer_synthetic_lesion_summary.txt")
print(summary(m))
sink()

cat("Generating Plot...\n")
p <- ggplot(trial_df, aes(x = Delta_Beta, y = P_Stay, color = Condition)) +
  geom_point(alpha = 0.05, position = position_jitter(width=0.02)) +
  geom_smooth(method = "lm", se = TRUE, size = 1.2) +
  facet_wrap(~ State) +
  scale_color_manual(values = c("Optimized" = "#d95f02", "Lesioned" = "#1b9e77")) +
  theme_minimal(base_size = 14) +
  labs(
    title = "Trial-Level Effect of Thalamic Lesion on Perseveration",
    subtitle = "Interaction between task state and magnitude of beta_thal ablation",
    x = bquote("Original Thalamic Rigidity (" ~ beta[thal] ~ ")"),
    y = "P(Stay | Previous Loss)"
  ) +
  theme(panel.spacing = unit(2, "lines"))

ggsave("results/Fig20B_LMER_Interaction.png", plot = p, width = 8, height = 5, dpi = 300)
cat("Done.\n")
