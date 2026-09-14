library(Rcpp)
library(dplyr)
sourceCpp("src/r/sim_bio_probes.cpp")

d_emp <- read.csv("data/behavioral_compilate.csv")
post_df <- readRDS("results/empirical_posteriors.rds")
params <- t(sapply(post_df[['bio_dist']][['traces']], function(x) exp(apply(x, 3, median))))

s_name <- unique(d_emp$participant_id)[1]
idx_s <- which(d_emp$participant_id == s_name)
choices_s <- d_emp$Resp[idx_s] - 1
ITI_s <- d_emp$ITI[idx_s]
F_s <- d_emp$F[idx_s]
X_s <- matrix(0, nrow=length(idx_s), ncol=2)
for(i in 1:length(idx_s)) X_s[i, d_emp$Resp[idx_s][i]] <- 1

s_idx <- which(names(post_df[['bio_dist']][['traces']]) == s_name)
W_gen <- post_df[['bio_dist']][['W_gen']][s_idx,,]
W_ach1 <- post_df[['bio_dist']][['W_ach1']][s_idx,,]
W_ach2 <- post_df[['bio_dist']][['W_ach2']][s_idx,,]
W_thal <- post_df[['bio_dist']][['W_thal']][s_idx,,]
Pi_s <- post_df[['bio_dist']][['Pi']][s_idx,]
opt_p <- params[s_idx, ]

# 1. Intact Traces
tr_opt <- simulate_bio_probes(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s,
                              opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1)
df_opt <- data.frame(Choice = choices_s, tr_opt$mu)
glm_opt <- glm(Choice ~ ., data = df_opt, family = binomial(link="logit"))
p1_opt <- predict(glm_opt, type="response")

# 2. Lesioned Traces (Evaluated on Intact GLM)
tr_les <- simulate_bio_probes(length(idx_s), X_s, ITI_s, W_gen, W_ach1, W_ach2, W_thal, Pi_s,
                              opt_p[1], opt_p[2], 0.0, opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0.1, 0.1)
df_les <- data.frame(Choice = choices_s, tr_les$mu)
p1_les_intact_glm <- predict(glm_opt, newdata = df_les, type="response")

# 3. Lesioned Traces (Evaluated on their OWN refitted GLM)
glm_les <- glm(Choice ~ ., data = df_les, family = binomial(link="logit"))
p1_les_new_glm <- predict(glm_les, type="response")

# Compare standard deviations of P1
cat("SD P1 Opt: ", sd(p1_opt), "\n")
cat("SD P1 Les (Intact GLM): ", sd(p1_les_intact_glm), "\n")
cat("SD P1 Les (Refit GLM): ", sd(p1_les_new_glm), "\n")

# Check if the refit restores the policy (does it stay near 0.5 or widen?)
cat("Mean P1 Les (Intact GLM): ", mean(p1_les_intact_glm), "\n")
cat("Mean P1 Les (Refit GLM): ", mean(p1_les_new_glm), "\n")
