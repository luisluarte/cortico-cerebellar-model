library(Rcpp)
library(RcppArmadillo)
library(ggplot2)
library(dplyr)
library(tidyr)
library(glmmTMB)
library(emmeans)

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

set.seed(42)
input_dim <- ncol(X_all)
W_gen <- matrix(rnorm(input_dim * 32, 0, 1/sqrt(32)), nrow=input_dim, ncol=32)
W_ach1 <- matrix(rnorm(2000 * 32, 0, 1/sqrt(32)), nrow=2000, ncol=32)
mask1 <- matrix(rbinom(2000 * 32, 1, 0.1), nrow=2000, ncol=32)
W_ach1 <- W_ach1 * mask1
W_ach2 <- matrix(rnorm(1000 * 2000, 0, 1/sqrt(2000)), nrow=1000, ncol=2000)
W_thal <- matrix(rnorm(32 * 1000, 0, 1/sqrt(1000)), nrow=32, ncol=1000)

agg_df <- data.frame()

for(s_idx in 1:length(unique_subjs)) {
  s_num <- unique_subjs[s_idx]
  s_name <- factor_ids[s_num]
  
  idx_s <- which(subjs == s_num)
  X_s <- X_all[idx_s, , drop=FALSE]
  ITI_s <- ITI_all[idx_s]
  Pi_s <- matrix(1, nrow=length(idx_s), ncol=input_dim)
  choices_s <- labels[idx_s]
  
  d_emp <- df_emp[df_emp$participant_id == s_name, ] %>% 
           filter(Resp %in% c(1, 2) & ((ttr - ttp) / 1000) > 0.1)
  
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
  
  base_traces_opt <- simulate_bio_probes(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s, 
                                     opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1)
  
  df_glm <- data.frame(Choice = choices_s, base_traces_opt$mu)
  suppressWarnings({ logit_model <- glm(Choice ~ ., data = df_glm, family = binomial(link="logit")) })
  p_choose1_opt <- predict(logit_model, type="response")
  prev_choice <- c(NA, choices_s[1:(length(choices_s)-1)])
  p_pers_opt <- ifelse(prev_choice == 1, p_choose1_opt, 1 - p_choose1_opt)
  
  base_traces_les <- simulate_bio_probes(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s, 
                                     opt_p[1], opt_p[2], 0.0, opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1)
  
  df_glm_les <- data.frame(Choice = choices_s, base_traces_les$mu)
  p_choose1_les <- predict(logit_model, newdata = df_glm_les, type="response")
  p_pers_les <- ifelse(prev_choice == 1, p_choose1_les, 1 - p_choose1_les)
  
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
agg_df$Prop <- agg_df$Perseverations / (agg_df$Perseverations + agg_df$Non_Perseverations)

cat("Fitting GLMM...\n")
m <- glmmTMB(cbind(Perseverations, Non_Perseverations) ~ Condition * State * Delta_Beta + (1 | Subject), 
             family = binomial, data = agg_df)

cat("Computing emmeans...\n")
sink("results/emmeans_lesion_summary.txt")
cat("=== 1. MARGINAL MEANS OF PERSEVERATION BY CONDITION & STATE ===\n")
emm <- emmeans(m, ~ Condition | State)
print(emm, type = "response")
cat("\n=== 2. PAIRWISE CONTRASTS: OPTIMIZED vs LESIONED (Are they significantly different?) ===\n")
print(pairs(emm))

cat("\n=== 3. MARGINAL SLOPES OF DELTA_BETA ON PERSEVERATION (Log-Odds Scale) ===\n")
emt <- emtrends(m, ~ Condition | State, var = "Delta_Beta")
print(emt)
cat("\n=== 4. PAIRWISE CONTRASTS OF SLOPES (Does ablation magnitude affect Opt/Les conditions differently?) ===\n")
print(pairs(emt))
sink()

cat("Generating Plot 2...\n")
wide_df <- agg_df %>% 
  select(Subject, State, Condition, Prop, Delta_Beta) %>% 
  pivot_wider(names_from = Condition, values_from = Prop) %>%
  mutate(Delta_Prop = Optimized - Lesioned)

p2 <- ggplot(wide_df, aes(x = Delta_Beta, y = Delta_Prop, color = State, fill = State)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_smooth(method = "lm", alpha = 0.2, linewidth=1.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  theme_minimal(base_size = 14) +
  scale_color_manual(values = c("Stable" = "#377eb8", "Reversal" = "#e41a1c")) +
  scale_fill_manual(values = c("Stable" = "#377eb8", "Reversal" = "#e41a1c")) +
  labs(
    title = "Ablation Magnitude vs. Intact Perseveration",
    subtitle = "How structural beta_thal dictates state-dependent behavioral inertia",
    x = bquote("Magnitude of Ablation (" ~ Delta ~ beta[thal] ~ ")"),
    y = expression(Delta ~ "P(Stay | Loss) [Optimized - Lesioned]")
  )

ggsave("results/Fig20E_Delta_Slopes.png", plot = p2, width = 7, height = 5, dpi = 300)
cat("Done.\n")
