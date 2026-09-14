fit <- readRDS("results/brms_global_parametric.rds")
df <- fit$data

subj_df <- df[!duplicated(df$Subject), ]
beta_thal <- exp(subj_df$log_Delta_Beta)

cat("--- Beta_Thal (From BRMS Model Data) Summary ---\n")
cat(sprintf("N Subjects : %d\n", length(beta_thal)))
cat(sprintf("Min        : %.3f\n", min(beta_thal)))
cat(sprintf("25%% (Q1)   : %.3f\n", quantile(beta_thal, 0.25)))
cat(sprintf("Median     : %.3f\n", median(beta_thal)))
cat(sprintf("Mean       : %.3f\n", mean(beta_thal)))
cat(sprintf("75%% (Q3)   : %.3f\n", quantile(beta_thal, 0.75)))
cat(sprintf("Max        : %.3f\n", max(beta_thal)))
