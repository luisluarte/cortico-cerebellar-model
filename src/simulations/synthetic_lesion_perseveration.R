library(Rcpp)
library(RcppArmadillo)
library(ggplot2)
library(dplyr)
library(tidyr)
library(ggpubr)

sourceCpp("src/r/sim_bio_probes.cpp")
sourceCpp("src/r/sim_agent.cpp")

cat("Loading Targets and Posteriors...\n")
targets <- readRDS("src/r/distillation_targets_v2.rds")
X_all <- targets$X
ITI_all <- targets$ITI / max(targets$ITI)
subjs <- targets$subjs
unique_subjs <- unique(subjs)

post_df <- readRDS("results/factorial_fixed_10k_A.rds")
# Use the median of posterior draws
params <- t(sapply(post_df[['bio_dist']][['traces']], function(x) exp(apply(x, 3, median))))
colnames(params) <- c('alpha_pc', 'lambda_pc', 'beta_thal', 'kappa_cf', 'alpha_gran', 'beta_gran', 'sigma2')

# Setup Topologies
set.seed(42)
input_dim <- ncol(X_all)
W_gen <- matrix(rnorm(input_dim * 32, 0, 1/sqrt(32)), nrow=input_dim, ncol=32)
W_ach1 <- matrix(rnorm(2000 * 32, 0, 1/sqrt(32)), nrow=2000, ncol=32)
mask1 <- matrix(rbinom(2000 * 32, 1, 0.1), nrow=2000, ncol=32)
W_ach1 <- W_ach1 * mask1
W_ach2 <- matrix(rnorm(1000 * 2000, 0, 1/sqrt(2000)), nrow=1000, ncol=2000)
W_thal <- matrix(rnorm(32 * 1000, 0, 1/sqrt(1000)), nrow=32, ncol=1000)

calc_perseveration <- function(choices) {
  if(length(choices) < 2) return(0)
  mean(choices[2:length(choices)] == choices[1:(length(choices)-1)])
}

arm_probs <- c(0.75, 0.25, 0.6, 0.4, 0.8, 0.2, 0.5, 0.5)

res_df <- data.frame()

cat("Running Synthetic Experiments...\n")
for(s_idx in 1:length(unique_subjs)) {
  s_name <- unique_subjs[s_idx]
  idx_s <- which(subjs == s_name)
  X_s <- X_all[idx_s, , drop=FALSE]
  ITI_s <- ITI_all[idx_s]
  Pi_s <- matrix(1, nrow=length(idx_s), ncol=input_dim)
  
  opt_p <- params[s_idx, ]
  
  # Step 1: Open-loop to get W_readout
  base_traces <- simulate_bio_probes(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s, 
                                     opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1)
  
  s_choices <- targets$labels[idx_s]
  df_glm <- data.frame(Choice = s_choices, base_traces$mu)
  
  suppressWarnings({
    logit_model <- glm(Choice ~ ., data = df_glm, family = binomial(link="logit"))
  })
  
  W_readout <- coef(logit_model)[-1]
  W_readout[is.na(W_readout)] <- 0
  intercept <- coef(logit_model)[1]
  if(is.na(intercept)) intercept <- 0
  
  # Step 2: Simulate Closed-Loop with OPTIMIZED beta_thal
  sim_opt <- simulate_agent_behavior(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s,
                                     opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1,
                                     W_readout, intercept, arm_probs)
  
  pers_opt <- calc_perseveration(sim_opt$choices)
  
  # Step 3: Simulate Closed-Loop with LESIONED beta_thal (= 0)
  sim_les <- simulate_agent_behavior(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s,
                                     opt_p[1], opt_p[2], 0.0, opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1,
                                     W_readout, intercept, arm_probs)
                                     
  pers_les <- calc_perseveration(sim_les$choices)
  
  res_df <- rbind(res_df, data.frame(
    Subject = s_name,
    Condition = "Optimized",
    Perseveration = pers_opt,
    Beta = opt_p[3]
  ))
  
  res_df <- rbind(res_df, data.frame(
    Subject = s_name,
    Condition = "Lesioned (Beta=0)",
    Perseveration = pers_les,
    Beta = 0.0
  ))
  
  if (s_idx %% 10 == 0) cat(sprintf("Processed %d / %d participants\n", s_idx, length(unique_subjs)))
}

cat("Generating Plot...\n")
res_df$Condition <- factor(res_df$Condition, levels = c("Optimized", "Lesioned (Beta=0)"))

p <- ggplot(res_df, aes(x = Condition, y = Perseveration)) +
  geom_line(aes(group = Subject), alpha = 0.3, color = "gray50") +
  geom_point(aes(color = Condition), size = 2, alpha = 0.7) +
  geom_boxplot(aes(fill = Condition), alpha = 0.3, width = 0.2, outlier.shape = NA) +
  scale_color_manual(values = c("Optimized" = "#d95f02", "Lesioned (Beta=0)" = "#1b9e77")) +
  scale_fill_manual(values = c("Optimized" = "#d95f02", "Lesioned (Beta=0)" = "#1b9e77")) +
  theme_minimal(base_size = 14) +
  labs(
    title = "Synthetic Lesion of Thalamic Feedback",
    subtitle = "Perseveration Index with Individual Posterior vs. Lesioned (β=0)",
    x = "Network State",
    y = "Perseveration Index P(Choice_t == Choice_{t-1})"
  ) +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank()
  )

ggsave("results/Fig19_Synthetic_Lesion_Perseveration.png", plot = p, width = 7, height = 6, dpi = 300)
cat("Done. Saved to results/Fig19_Synthetic_Lesion_Perseveration.png\n")

w_test <- wilcox.test(
  res_df$Perseveration[res_df$Condition == "Optimized"], 
  res_df$Perseveration[res_df$Condition == "Lesioned (Beta=0)"], 
  paired = TRUE
)
cat("\nWilcoxon Signed-Rank Test:\n")
print(w_test)
