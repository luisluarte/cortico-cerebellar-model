
res_bio <- readRDS("fast_res_bio.rds")
df <- as.data.frame(res_bio$par[res_bio$pareto.optimal, , drop=FALSE])
colnames(df) <- c("alpha_pc", "lambda_pc", "beta_thal", "kappa_cf", "alpha_gran", "beta_gran", "sigma2_diff", "gamma")

df <- df[order(df$gamma), ]
n <- nrow(df)
q1 <- df[max(1, floor(n * 0.25)), ]
q2 <- df[max(1, floor(n * 0.50)), ]
q3 <- df[max(1, floor(n * 0.75)), ]

cat("\n=== CORTICO-CEREBELLAR PARAMETERS (FULL 50-SUBJECT RUN) ===\n")
cat("Q1 (25th Percentile Gamma):\n")
print(q1)
cat("\nQ2 (Median Gamma):\n")
print(q2)
cat("\nQ3 (75th Percentile Gamma):\n")
print(q3)
