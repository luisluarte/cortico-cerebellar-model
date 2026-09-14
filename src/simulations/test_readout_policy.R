library(Rcpp)
library(dplyr)
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

input_dim <- ncol(X_all)
set.seed(42)
W_gen <- matrix(rnorm(input_dim * 32, 0, 1/sqrt(32)), nrow=input_dim, ncol=32)
W_ach1 <- matrix(rnorm(2000 * 32, 0, 1/sqrt(32)), nrow=2000, ncol=32)
mask1 <- matrix(rbinom(2000 * 32, 1, 0.1), nrow=2000, ncol=32)
W_ach1 <- W_ach1 * mask1
W_ach2 <- matrix(rnorm(1000 * 2000, 0, 1/sqrt(2000)), nrow=1000, ncol=2000)
W_thal <- matrix(rnorm(32 * 1000, 0, 1/sqrt(1000)), nrow=32, ncol=1000)

s_idx <- 1
s_num <- unique_subjs[s_idx]
idx_s <- which(subjs == s_num)
X_s <- X_all[idx_s, , drop=FALSE]
ITI_s <- ITI_all[idx_s]
Pi_s <- matrix(1, nrow=length(idx_s), ncol=input_dim)
choices_s <- labels[idx_s]
F_s <- targets$F[idx_s]
opt_p <- params[s_idx, ]

tr_opt <- simulate_bio_probes(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s,
                              opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1)
df_opt <- data.frame(Choice = choices_s, tr_opt$mu)
glm_opt <- suppressWarnings(glm(Choice ~ ., data = df_opt, family = binomial(link="logit")))
p1_opt <- predict(glm_opt, type="response")

tr_les <- simulate_bio_probes(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s,
                              opt_p[1], opt_p[2], 0.0, opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1)
df_les <- data.frame(Choice = choices_s, tr_les$mu)
p1_les_intact_glm <- suppressWarnings(predict(glm_opt, newdata = df_les, type="response"))

glm_les <- suppressWarnings(glm(Choice ~ ., data = df_les, family = binomial(link="logit")))
p1_les_new_glm <- suppressWarnings(predict(glm_les, type="response"))

# Create data frame to analyze policy
res <- data.frame(
  F = F_s,
  Opt = p1_opt,
  Les_IntactGLM = p1_les_intact_glm,
  Les_RefitGLM = p1_les_new_glm
)

cat("--- Policy for Refit Lesion ---\n")
print(res %>% group_by(F) %>% summarize(
  P_Switch_Opt = mean(Opt),
  P_Switch_Les_Old = mean(Les_IntactGLM),
  P_Switch_Les_New = mean(Les_RefitGLM)
))
