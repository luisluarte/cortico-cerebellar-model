library(Rcpp)
library(RcppArmadillo)
library(ggplot2)
library(dplyr)
library(tidyr)
library(ggpubr)

# Ensure glmmTMB and emmeans are available
if(!require(glmmTMB)) install.packages("glmmTMB", repos="https://cloud.r-project.org")
if(!require(emmeans)) install.packages("emmeans", repos="https://cloud.r-project.org")
library(glmmTMB)
library(emmeans)

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

agg_df <- data.frame()

cat("Extracting Expected Counts for Beta-Binomial...\n")
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
  stable_window <- setdiff(1:nrow(d_emp), rev_window)
  
  opt_p <- params[s_idx, ]
  delta_beta <- opt_p[3]
  
  # OPTIMIZED
  base_traces_opt <- simulate_bio_probes(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s, 
                                     opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1)
  
  df_glm <- data.frame(Choice = choices_s, base_traces_opt$mu)
  suppressWarnings({
    logit_model <- glm(Choice ~ ., data = df_glm, family = binomial(link="logit"))
  })
  p_choose1_opt <- predict(logit_model, type="response")
  prev_choice <- c(NA, choices_s[1:(length(choices_s)-1)])
  p_pers_opt <- ifelse(prev_choice == 1, p_choose1_opt, 1 - p_choose1_opt)
  
  # LESIONED
  base_traces_les <- simulate_bio_probes(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s, 
                                     opt_p[1], opt_p[2], 0.0, opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1)
  
  df_glm_les <- data.frame(Choice = choices_s, base_traces_les$mu)
  p_choose1_les <- predict(logit_model, newdata = df_glm_les, type="response")
  p_pers_les <- ifelse(prev_choice == 1, p_choose1_les, 1 - p_choose1_les)
  
  # AGGREGATE
  prev_reward <- c(NA, d_emp$F[1:(nrow(d_emp)-1)])
  loss_idx <- which(prev_reward == 0)
  
  loss_stable <- intersect(loss_idx, stable_window)
  loss_rev <- intersect(loss_idx, rev_window)
  
  add_to_agg <- function(state_name, trials, p_probs) {
    if(length(trials) > 0) {
       pers <- round(sum(p_probs[trials], na.rm=TRUE))
       non_pers <- length(trials) - pers
       return(data.frame(Subject=s_name, State=state_name, Perseverations=pers, Non_Perseverations=non_pers, Delta_Beta=delta_beta))
    }
    return(NULL)
  }
  
  opt_s <- add_to_agg("Stable", loss_stable, p_pers_opt)
  opt_r <- add_to_agg("Reversal", loss_rev, p_pers_opt)
  les_s <- add_to_agg("Stable", loss_stable, p_pers_les)
  les_r <- add_to_agg("Reversal", loss_rev, p_pers_les)
  
  if(!is.null(opt_s)) agg_df <- rbind(agg_df, cbind(opt_s, Condition="Optimized"))
  if(!is.null(opt_r)) agg_df <- rbind(agg_df, cbind(opt_r, Condition="Optimized"))
  if(!is.null(les_s)) agg_df <- rbind(agg_df, cbind(les_s, Condition="Lesioned"))
  if(!is.null(les_r)) agg_df <- rbind(agg_df, cbind(les_r, Condition="Lesioned"))
}

agg_df$Condition <- factor(agg_df$Condition, levels = c("Lesioned", "Optimized"))
agg_df$State <- factor(agg_df$State, levels = c("Stable", "Reversal"))
agg_df$Total <- agg_df$Perseverations + agg_df$Non_Perseverations
agg_df$Prop <- agg_df$Perseverations / agg_df$Total

cat("Running Beta-Binomial Model...\n")
m <- glmmTMB(cbind(Perseverations, Non_Perseverations) ~ Condition * State * Delta_Beta + (1 | Subject), 
             family = binomial, data = agg_df)

sink("results/glmmTMB_synthetic_lesion_summary.txt")
print(summary(m))
sink()

cat("Generating Plots...\n")
# Plot 1: Overall effect per condition
p1 <- ggplot(agg_df, aes(x = Condition, y = Prop, fill = Condition)) +
  geom_boxplot(alpha = 0.5, outlier.shape = NA, width = 0.5) +
  geom_point(aes(color = Condition), position = position_jitter(width=0.1), alpha = 0.5) +
  geom_line(aes(group = Subject), alpha = 0.1, color = "gray20") +
  facet_wrap(~ State) +
  scale_color_manual(values = c("Optimized" = "#d95f02", "Lesioned" = "#1b9e77")) +
  scale_fill_manual(values = c("Optimized" = "#d95f02", "Lesioned" = "#1b9e77")) +
  theme_minimal(base_size = 14) +
  labs(
    title = "Overall Lesion Effect on Perseveration",
    subtitle = "Aggregated Proportion of Perseverative Choices (Beta-Binomial)",
    y = "P(Stay | Previous Loss)", x = ""
  ) +
  theme(legend.position = "none")

# Plot 2: Estimate per condition as Delta_Beta increases
# Generate predicted marginal means
new_data <- expand.grid(
  Condition = c("Lesioned", "Optimized"),
  State = c("Stable", "Reversal"),
  Delta_Beta = seq(min(agg_df$Delta_Beta), max(agg_df$Delta_Beta), length.out = 100),
  Subject = NA
)

# Predict proportions using population level (re.form = NA)
new_data$Predicted_Prop <- predict(m, newdata = new_data, type = "response", re.form = NA)

p2 <- ggplot(agg_df, aes(x = Delta_Beta, y = Prop, color = Condition)) +
  geom_point(alpha = 0.2, position = position_jitter(width=0.01)) +
  geom_line(data = new_data, aes(x = Delta_Beta, y = Predicted_Prop, color = Condition), size = 1.5) +
  facet_wrap(~ State) +
  scale_color_manual(values = c("Optimized" = "#d95f02", "Lesioned" = "#1b9e77")) +
  theme_minimal(base_size = 14) +
  labs(
    title = "Perseveration Trajectories by Original Thalamic Rigidity",
    subtitle = "Predicted Beta-Binomial marginal effects",
    x = bquote("Original Thalamic Rigidity (" ~ beta[thal] ~ ")"),
    y = "Expected P(Stay | Previous Loss)"
  )

ggsave("results/Fig20C_GLMM_Overall.png", plot = p1, width = 8, height = 5, dpi = 300)
ggsave("results/Fig20D_GLMM_Slopes.png", plot = p2, width = 8, height = 5, dpi = 300)

cat("Done.\n")
