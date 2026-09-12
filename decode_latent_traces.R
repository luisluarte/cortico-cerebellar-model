library(Rcpp)
library(RcppArmadillo)
library(glmnet)
library(ggplot2)
library(dplyr)
library(tidyr)
library(zoo)

cat("Loading Simulation Core and Data...\n")
sourceCpp("src/r/sim_bio_probes.cpp")

targets <- readRDS("src/r/distillation_targets_v2.rds")
N_trials <- length(targets$labels)
X <- targets$X
ITI <- targets$ITI
ITI <- ITI / max(ITI)
lag_Reward <- targets$lag_reward
t_subjs <- targets$subjs

set.seed(42)
input_dim <- ncol(X)
W_gen <- matrix(rnorm(input_dim * 32, 0, 1/sqrt(32)), nrow=input_dim, ncol=32)
W_ach1 <- matrix(rnorm(2000 * 32, 0, 1/sqrt(32)), nrow=2000, ncol=32)
mask1 <- matrix(rbinom(2000 * 32, 1, 0.1), nrow=2000, ncol=32)
W_ach1 <- W_ach1 * mask1
W_ach2 <- matrix(rnorm(1000 * 2000, 0, 1/sqrt(2000)), nrow=1000, ncol=2000)
W_thal <- matrix(rnorm(32 * 1000, 0, 1/sqrt(1000)), nrow=32, ncol=1000)

Pi_mat <- matrix(1, nrow=N_trials, ncol=input_dim)
train_subjs <- unique(t_subjs)

cat("Fetching Optimal God Parameters...\n")
res <- readRDS("results/nsga2_pruning_results.rds")
opt_p <- res$par[1, ]

idx <- which(t_subjs == train_subjs[1])
subj_X <- X[idx, , drop=FALSE]
subj_ITI <- ITI[idx]
subj_Pi <- Pi_mat[idx, , drop=FALSE]
subj_reward <- lag_Reward[idx]

cat("Running Forward Probe Extraction...\n")
traces <- simulate_bio_probes(length(idx), subj_X, subj_ITI, W_gen, W_ach1, W_ach2, W_thal, subj_Pi, 
                              opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7],
                              opt_p[8], opt_p[9])

# Compute Rolling Targets
# Value: Moving average of past 10 rewards
val_target <- rollmean(subj_reward, k=10, fill=NA, align="right")

# Uncertainty: Entropy of rolling win rate
roll_win <- rollmean(subj_reward, k=10, fill=NA, align="right")
eps <- 1e-5
roll_win <- pmax(eps, pmin(1-eps, roll_win))
unc_target <- -(roll_win * log2(roll_win) + (1-roll_win) * log2(1-roll_win))

# Filter NAs
valid_idx <- which(!is.na(val_target) & !is.na(unc_target))
val_target <- val_target[valid_idx]
unc_target <- unc_target[valid_idx]

decode_layer <- function(layer_mat, target, layer_name, target_name) {
  X_mat <- layer_mat[valid_idx, ]
  # Use cross-validated Ridge Regression to prevent overfitting on high-dim layers
  cv_fit <- cv.glmnet(X_mat, target, alpha = 0, nfolds = 5)
  preds <- predict(cv_fit, newx = X_mat, s = "lambda.min")
  r2 <- cor(preds, target)^2
  return(data.frame(Layer = layer_name, Target = target_name, R2 = as.numeric(r2)))
}

cat("Training Linear Decoders on Layer Traces...\n")
res_df <- data.frame()

# Value Decoding
res_df <- rbind(res_df, decode_layer(traces$mu, val_target, "1. Cortex", "Expected Value"))
res_df <- rbind(res_df, decode_layer(traces$Z, val_target, "2. Granule", "Expected Value"))
res_df <- rbind(res_df, decode_layer(traces$P, val_target, "3. Purkinje", "Expected Value"))
res_df <- rbind(res_df, decode_layer(traces$D, val_target, "4. DCN", "Expected Value"))

# Uncertainty Decoding
res_df <- rbind(res_df, decode_layer(traces$mu, unc_target, "1. Cortex", "Reward Entropy"))
res_df <- rbind(res_df, decode_layer(traces$Z, unc_target, "2. Granule", "Reward Entropy"))
res_df <- rbind(res_df, decode_layer(traces$P, unc_target, "3. Purkinje", "Reward Entropy"))
res_df <- rbind(res_df, decode_layer(traces$D, unc_target, "4. DCN", "Reward Entropy"))

cat("Generating Representation Plot...\n")

p <- ggplot(res_df, aes(x = Layer, y = R2, fill = Target)) +
  geom_bar(stat = "identity", position = "dodge", color="black", alpha=0.9) +
  scale_fill_manual(values = c("Expected Value" = "#2a9d7a", "Reward Entropy" = "#7572b8")) +
  labs(title = "Latent Decoding Capacity of the Cortico-Cerebellar Loop",
       subtitle = "Cross-Validated Ridge Regression R² from Internal Layer Traces",
       x = "Biological Layer", y = "Decoding R²") +
  theme_minimal(base_size=14) +
  theme(legend.position = "bottom",
        strip.background = element_rect(fill = "white", color = "black"),
        panel.grid.major.x = element_blank())

ggsave("results/Fig5_Representation_Decoding.png", plot = p, width = 8, height = 5, dpi = 300)
saveRDS(res_df, "results/layer_decoding_results.rds")
cat("Finished! Plot saved to Fig5_Representation_Decoding.png\n")
