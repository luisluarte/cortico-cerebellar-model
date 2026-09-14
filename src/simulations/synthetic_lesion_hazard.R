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

hazard_df <- data.frame()
curve_df <- data.frame()

cat("Computing Reversal Hazard Rates...\n")
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
  
  # OPTIMIZED
  base_traces_opt <- simulate_bio_probes(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s, 
                                     opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1)
  df_glm <- data.frame(Choice = choices_s, base_traces_opt$mu)
  suppressWarnings({ logit_model <- glm(Choice ~ ., data = df_glm, family = binomial(link="logit")) })
  p_choose1_opt <- predict(logit_model, type="response")
  
  # LESIONED
  base_traces_les <- simulate_bio_probes(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s, 
                                     opt_p[1], opt_p[2], 0.0, opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1)
  df_glm_les <- data.frame(Choice = choices_s, base_traces_les$mu)
  p_choose1_les <- predict(logit_model, newdata = df_glm_les, type="response")
  
  # HAZARD RATE IDENTIFICATION
  prev_reward <- c(NA, d_emp$F[1:(nrow(d_emp)-1)])
  d_emp$diff_prob <- c(0, diff(d_emp$prob))
  rev_idx <- which(abs(d_emp$diff_prob) > 0.4)
  
  subj_opt_hazards <- matrix(NA, nrow=length(rev_idx), ncol=5)
  subj_les_hazards <- matrix(NA, nrow=length(rev_idx), ncol=5)
  
  for(k in seq_along(rev_idx)) {
    r <- rev_idx[k]
    # Find first loss on or after the reversal trial (up to 10 trials later)
    search_win <- r:min(r+10, length(prev_reward))
    losses <- search_win[which(prev_reward[search_win] == 0)]
    
    if(length(losses) > 0) {
      t_start <- losses[1]
      
      # Ensure we have a 5-trial window
      if(t_start + 4 <= length(choices_s)) {
        win <- t_start:(t_start+4)
        c_old <- choices_s[t_start - 1] # The favored arm that resulted in the first loss
        
        if(!is.na(c_old)) {
           # P(Switch) is P(Choice != c_old)
           p_switch_opt <- ifelse(c_old == 1, 1 - p_choose1_opt[win], p_choose1_opt[win])
           p_switch_les <- ifelse(c_old == 1, 1 - p_choose1_les[win], p_choose1_les[win])
           
           subj_opt_hazards[k, ] <- p_switch_opt
           subj_les_hazards[k, ] <- p_switch_les
        }
      }
    }
  }
  
  mean_opt_curve <- colMeans(subj_opt_hazards, na.rm=TRUE)
  mean_les_curve <- colMeans(subj_les_hazards, na.rm=TRUE)
  
  if(!any(is.na(mean_opt_curve))) {
      # Average Hazard Rate over the 5 trials
      hazard_df <- rbind(hazard_df, data.frame(Subject=s_name, Condition="Optimized", Hazard=mean(mean_opt_curve)))
      hazard_df <- rbind(hazard_df, data.frame(Subject=s_name, Condition="Lesioned", Hazard=mean(mean_les_curve)))
      
      # Step-by-step curve
      for(trial in 1:5) {
         curve_df <- rbind(curve_df, data.frame(Subject=s_name, Condition="Optimized", Trial=trial, Hazard=mean_opt_curve[trial]))
         curve_df <- rbind(curve_df, data.frame(Subject=s_name, Condition="Lesioned", Trial=trial, Hazard=mean_les_curve[trial]))
      }
  }
}

hazard_df$Condition <- factor(hazard_df$Condition, levels=c("Optimized", "Lesioned"))
curve_df$Condition <- factor(curve_df$Condition, levels=c("Optimized", "Lesioned"))

cat("Statistical Testing...\n")
w_test <- wilcox.test(
  hazard_df$Hazard[hazard_df$Condition == "Optimized"],
  hazard_df$Hazard[hazard_df$Condition == "Lesioned"],
  paired = TRUE
)
sink("results/hazard_reversal_stats.txt")
cat("Wilcoxon Paired Test for Average Reversal Hazard Rate (Optimized vs Lesioned):\n")
print(w_test)
sink()

cat("Plotting...\n")
p_box <- ggplot(hazard_df, aes(x = Condition, y = Hazard, fill=Condition)) +
   geom_boxplot(alpha=0.5, outlier.shape=NA, width=0.5) +
   geom_point(position=position_jitter(width=0.1), alpha=0.5, aes(color=Condition)) +
   geom_line(aes(group=Subject), alpha=0.2, color="gray50") +
   scale_fill_manual(values = c("Optimized" = "#d95f02", "Lesioned" = "#1b9e77")) +
   scale_color_manual(values = c("Optimized" = "#d95f02", "Lesioned" = "#1b9e77")) +
   theme_minimal(base_size=14) +
   labs(title="Reversal Hazard Rate (5-Trial Window)",
        subtitle="Probability of switching from obsolete arm after first loss",
        y="Average P(Switch | Obsolete Arm)", x="") +
   theme(legend.position="none")

p_curve <- ggplot(curve_df, aes(x = Trial, y = Hazard, color=Condition, fill=Condition)) +
   stat_summary(fun = mean, geom="line", linewidth=1.5) +
   stat_summary(fun.data = mean_se, geom="ribbon", alpha=0.2, color=NA) +
   scale_color_manual(values = c("Optimized" = "#d95f02", "Lesioned" = "#1b9e77")) +
   scale_fill_manual(values = c("Optimized" = "#d95f02", "Lesioned" = "#1b9e77")) +
   theme_minimal(base_size=14) +
   labs(title="Hazard Rate Dynamics",
        subtitle="Timecourse of attractor abandonment over the 5 trials",
        x="Trials Since First Loss",
        y="P(Switch | Obsolete Arm)") +
   theme(legend.position="bottom")

ggsave("results/Fig21A_Hazard_Boxplot.png", plot=p_box, width=6, height=5, dpi=300)
ggsave("results/Fig21B_Hazard_Curve.png", plot=p_curve, width=6, height=5, dpi=300)

cat("Done.\n")
