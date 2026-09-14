df <- readRDS("results/reversal_df.rds")

# Extract 1 row per subject to get true biological distribution
subj_df <- df[!duplicated(df$Subject), ]
beta_thal <- subj_df$Delta_Beta

cat("--- Native Beta_Thal (Structural Integrity) Summary ---\n")
cat(sprintf("N Subjects : %d\n", length(beta_thal)))
cat(sprintf("Min        : %.3f\n", min(beta_thal)))
cat(sprintf("25%% (Q1)   : %.3f\n", quantile(beta_thal, 0.25)))
cat(sprintf("Median     : %.3f\n", median(beta_thal)))
cat(sprintf("Mean       : %.3f\n", mean(beta_thal)))
cat(sprintf("75%% (Q3)   : %.3f\n", quantile(beta_thal, 0.75)))
cat(sprintf("Max        : %.3f\n", max(beta_thal)))
