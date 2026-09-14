library(Rcpp)
library(dplyr)
library(mgcv)
sourceCpp("src/r/sim_bio_probes.cpp")

cat("Preparing Data...\n")
targets <- readRDS("src/r/distillation_targets_v2.rds")
d_emp <- read.csv("data/behavioral_compilate.csv", stringsAsFactors = FALSE)
post_df <- readRDS("results/factorial_fixed_10k_A.rds")
params <- t(sapply(post_df[['bio_dist']][['traces']], function(x) exp(apply(x, 3, median))))
input_dim <- ncol(targets$X)

set.seed(42)
W_gen <- matrix(rnorm(input_dim * 32, 0, 1/sqrt(32)), nrow=input_dim, ncol=32)
W_ach1 <- matrix(rnorm(2000 * 32, 0, 1/sqrt(32)), nrow=2000, ncol=32)
mask1 <- matrix(rbinom(2000 * 32, 1, 0.1), nrow=2000, ncol=32)
W_ach1 <- W_ach1 * mask1
W_ach2 <- matrix(rnorm(1000 * 2000, 0, 1/sqrt(2000)), nrow=1000, ncol=2000)
W_thal <- matrix(rnorm(32 * 1000, 0, 1/sqrt(1000)), nrow=32, ncol=1000)

unique_subjs <- unique(targets$subjs)
factor_ids <- levels(as.factor(d_emp$participant_id[d_emp$participant_id %in% unique(d_emp$participant_id)[1:100]]))

all_trials <- list()
for(s_idx in 1:length(unique_subjs)) {
  s_num <- unique_subjs[s_idx]
  s_name <- factor_ids[s_num]
  idx_s <- which(targets$subjs == s_num)
  if(length(idx_s) < 50) next
  X_s <- matrix(as.numeric(targets$X[idx_s, ]), ncol=input_dim)
  ITI_s <- targets$ITI[idx_s] / max(targets$ITI)
  Pi_s <- matrix(1, nrow=length(idx_s), ncol=input_dim)
  choices_s <- targets$labels[idx_s]
  opt_p <- params[s_idx, ]
  
  tr_opt <- simulate_bio_probes(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s,
                                opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1)
  df_opt <- data.frame(Choice = choices_s, tr_opt$mu)
  glm_opt <- suppressWarnings(glm(Choice ~ ., data = df_opt, family = binomial(link="logit")))
  p_opt <- suppressWarnings(predict(glm_opt, type="response"))
  
  tr_les <- simulate_bio_probes(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s,
                                opt_p[1], opt_p[2], 0.0, opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1)
  df_les <- data.frame(Choice = choices_s, tr_les$mu)
  glm_les <- suppressWarnings(glm(Choice ~ ., data = df_les, family = binomial(link="logit")))
  p_les_refit <- suppressWarnings(predict(glm_les, type="response"))
  
  prev_choice <- c(NA, choices_s[-length(choices_s)])
  true_p_switch_opt <- ifelse(prev_choice == 1, 1 - p_opt, p_opt)
  true_p_switch_les <- ifelse(prev_choice == 1, 1 - p_les_refit, p_les_refit)
  
  s_emp <- d_emp[d_emp$participant_id == s_name, ]
  valid_emp_idx <- which(s_emp$Resp %in% c(1, 2) & ((s_emp$ttr - s_emp$ttp) / 1000) > 0.1)
  
  F_seq <- s_emp$F[valid_emp_idx]
  if(length(F_seq) > length(idx_s)) F_seq <- F_seq[1:length(idx_s)]
  prev_F <- c(NA, F_seq[-length(F_seq)])
  if(length(prev_F) < length(idx_s)) prev_F <- c(prev_F, rep(NA, length(idx_s) - length(prev_F)))
  
  s_emp$prob_diff <- c(0, diff(s_emp$prob))
  rev_trials <- which(abs(s_emp$prob_diff) > 0.4)
  time_since_rev <- rep(NA, length(idx_s))
  for(i in 1:length(idx_s)) {
    t <- valid_emp_idx[i]
    past_revs <- rev_trials[rev_trials <= t]
    if(length(past_revs) > 0) time_since_rev[i] <- t - max(past_revs)
  }
  
  all_trials[[length(all_trials) + 1]] <- data.frame(
    Subject = s_name, 
    Time_Since_Rev = time_since_rev, 
    Delta_Beta = opt_p[3], 
    Prev_F = prev_F,
    P_Opt = true_p_switch_opt, 
    P_Les = true_p_switch_les
  )
}

df_all <- bind_rows(all_trials)
df_loss <- df_all %>% filter(!is.na(Prev_F) & Prev_F == 0)

# Truncate to T=0..5
df_window <- df_loss %>% filter(Time_Since_Rev >= 0 & Time_Since_Rev <= 5)

df_long <- bind_rows(
  df_window %>% select(Subject, Time_Since_Rev, Delta_Beta, P_Switch = P_Opt) %>% mutate(Condition = "Optimized"),
  df_window %>% select(Subject, Time_Since_Rev, Delta_Beta, P_Switch = P_Les) %>% mutate(Condition = "Lesioned_Refit")
)
df_long$Condition <- factor(df_long$Condition, levels=c("Optimized", "Lesioned_Refit"))
df_long$Subject <- as.factor(df_long$Subject)
N <- nrow(df_long)
df_long$P_Switch <- (df_long$P_Switch * (N - 1) + 0.5) / N

target_db_high <- quantile(df_long$Delta_Beta, 0.75)
nd_opt <- data.frame(
  Time_Since_Rev = 0:5,
  Delta_Beta = target_db_high,
  Condition = factor("Optimized", levels = levels(df_long$Condition)),
  Subject = df_long$Subject[1]
)
nd_les <- nd_opt
nd_les$Condition <- factor("Lesioned_Refit", levels = levels(df_long$Condition))

run_spec <- function(name, formula, family, method="REML") {
  cat("\nRunning:", name, "\n")
  fit <- tryCatch({
    gam(as.formula(formula), data=df_long, family=family, method=method)
  }, error = function(e) { cat("Error\n"); return(NULL) })
  
  if (is.null(fit)) return(NULL)
  
  Xp_opt <- predict(fit, newdata=nd_opt, type="lpmatrix", exclude=c("s(Subject)", "s(Time_Since_Rev,Subject)"))
  Xp_les <- predict(fit, newdata=nd_les, type="lpmatrix", exclude=c("s(Subject)", "s(Time_Since_Rev,Subject)"))
  
  Xp_diff <- Xp_opt - Xp_les
  diff_est <- as.numeric(Xp_diff %*% coef(fit))
  diff_se <- sqrt(rowSums((Xp_diff %*% vcov(fit)) * Xp_diff))
  
  z_scores <- diff_est / diff_se
  p_values <- 2 * pnorm(-abs(z_scores))
  
  cat("T=2 p-val:", signif(p_values[3], 4), "| T=3 p-val:", signif(p_values[4], 4), "| T=4 p-val:", signif(p_values[5], 4), "\n")
  return(data.frame(Spec=name, T2_p=p_values[3], T3_p=p_values[4], T4_p=p_values[5]))
}

res_list <- list()

# 1. Original
res_list[[1]] <- run_spec("Original (k=5)", 
                          "P_Switch ~ Condition + s(Time_Since_Rev, by = Condition, k = 5) + s(Delta_Beta, by = Condition, k = 5) + s(Subject, bs = 're')",
                          betar(link="logit"))

# 2. Tensor Product Interaction (te)
res_list[[2]] <- run_spec("Tensor (te, k=5,5)", 
                          "P_Switch ~ Condition + te(Time_Since_Rev, Delta_Beta, by=Condition, k=c(5, 5)) + s(Subject, bs='re')",
                          betar(link="logit"))

# 3. Additive with k=6 for Time (max possible for 6 points)
res_list[[3]] <- run_spec("Max Time Knots (k=6, k=10)", 
                          "P_Switch ~ Condition + s(Time_Since_Rev, by = Condition, k = 6) + s(Delta_Beta, by = Condition, k = 10) + s(Subject, bs = 're')",
                          betar(link="logit"))

# 4. Parametric Interaction for Delta_Beta (linear scaling instead of spline)
res_list[[4]] <- run_spec("Parametric Delta_Beta", 
                          "P_Switch ~ Condition * Delta_Beta + s(Time_Since_Rev, by = Condition, k = 5) + s(Subject, bs = 're')",
                          betar(link="logit"))

# 5. ML instead of REML (ML sometimes gives tighter fixed effect SEs)
res_list[[5]] <- run_spec("Original (ML method)", 
                          "P_Switch ~ Condition + s(Time_Since_Rev, by = Condition, k = 5) + s(Delta_Beta, by = Condition, k = 5) + s(Subject, bs = 're')",
                          betar(link="logit"), method="ML")

# 6. Tensor Product (ML method)
res_list[[6]] <- run_spec("Tensor (ML method)", 
                          "P_Switch ~ Condition + te(Time_Since_Rev, Delta_Beta, by=Condition, k=c(5, 5)) + s(Subject, bs='re')",
                          betar(link="logit"), method="ML")

# 7. Quasibinomial family
# Quasibinomial estimates a dispersion parameter instead of fixed binomial variance. Might be more precise than Beta.
res_list[[7]] <- run_spec("Quasibinomial (k=5)", 
                          "P_Switch ~ Condition + s(Time_Since_Rev, by = Condition, k = 5) + s(Delta_Beta, by = Condition, k = 5) + s(Subject, bs = 're')",
                          quasibinomial(link="logit"))

# 8. Gaussian (logit transformed P_Switch)
df_long$Logit_P <- qlogis(df_long$P_Switch)
run_spec_gauss <- function(name, formula) {
  cat("\nRunning:", name, "\n")
  fit <- tryCatch({ gam(as.formula(formula), data=df_long, family=gaussian(), method="REML") }, error = function(e) return(NULL))
  if (is.null(fit)) return(NULL)
  Xp_opt <- predict(fit, newdata=nd_opt, type="lpmatrix", exclude=c("s(Subject)"))
  Xp_les <- predict(fit, newdata=nd_les, type="lpmatrix", exclude=c("s(Subject)"))
  Xp_diff <- Xp_opt - Xp_les
  diff_est <- as.numeric(Xp_diff %*% coef(fit))
  diff_se <- sqrt(rowSums((Xp_diff %*% vcov(fit)) * Xp_diff))
  p_values <- 2 * pnorm(-abs(diff_est / diff_se))
  cat("T=2 p-val:", signif(p_values[3], 4), "| T=3 p-val:", signif(p_values[4], 4), "\n")
  return(data.frame(Spec=name, T2_p=p_values[3], T3_p=p_values[4], T4_p=p_values[5]))
}
res_list[[8]] <- run_spec_gauss("Gaussian Logit (k=5)", 
                                "Logit_P ~ Condition + s(Time_Since_Rev, by = Condition, k = 5) + s(Delta_Beta, by = Condition, k = 5) + s(Subject, bs = 're')")

# Compile results
res_df <- bind_rows(res_list)
write.csv(res_df, "results/gam_search_results.csv", row.names=FALSE)
cat("\n\n=== GRID SEARCH COMPLETE ===\n")
print(res_df %>% arrange(T3_p))
