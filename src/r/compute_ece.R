
local_lib <- Sys.getenv("R_LIBS_USER")
.libPaths(c(local_lib, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"))

suppressPackageStartupMessages({
    library(torch)
    library(jsonlite)
    library(dplyr)
})

cat("Loading data...\n")
stan_data <- read_json("../../data/stan_data.json", simplifyVector = TRUE)
rnn_phase1_subjs <- readRDS("rnn_phase1_subjects.rds")

# Ensure CUDA is available
device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
cat(sprintf("running on device: %s\n", as.character(device)))

# Process features identically to rnn_teacher.R
N_trials <- stan_data$N
min_RT <- min(stan_data$RT)

Ch <- numeric(N_trials)
for (i in 1:N_trials) Ch[i] <- ifelse(stan_data$Resp[i] == 1, stan_data$Bd1[i], stan_data$Bd2[i])

Switch <- numeric(N_trials)
lag_Reward <- numeric(N_trials)
lag_Ch <- numeric(N_trials)
lag_RT <- numeric(N_trials)
lag_Resp <- numeric(N_trials)

current_subj <- -1
for (i in 1:N_trials) {
    if (stan_data$subj[i] != current_subj) {
        Switch[i] <- 0; lag_Reward[i] <- 0; lag_Ch[i] <- 0; lag_RT[i] <- 0; lag_Resp[i] <- 0
        current_subj <- stan_data$subj[i]
    } else {
        Switch[i] <- ifelse(stan_data$Resp[i] != stan_data$Resp[i-1], 1, 0)
        lag_Reward[i] <- stan_data$Reward[i-1]; lag_Ch[i] <- Ch[i-1]; lag_RT[i] <- stan_data$RT[i-1]; lag_Resp[i] <- ifelse(stan_data$Resp[i-1] == 1, 1, 0)
    }
}

X_Bd1 <- matrix(0, nrow = N_trials, ncol = 8)
X_Bd2 <- matrix(0, nrow = N_trials, ncol = 8)
X_lag_Ch <- matrix(0, nrow = N_trials, ncol = 8)
for (i in 1:N_trials) {
    X_Bd1[i, stan_data$Bd1[i]] <- 1; X_Bd2[i, stan_data$Bd2[i]] <- 1
    if (lag_Ch[i] > 0) X_lag_Ch[i, lag_Ch[i]] <- 1
}
X <- cbind(X_Bd1, X_Bd2, lag_Reward, lag_RT, X_lag_Ch, lag_Resp)

# Model definition
SpatialRNN <- nn_module(
    "SpatialRNN",
    initialize = function(input_dim, hidden_dim, K) {
        self$gru <- nn_gru(input_dim, hidden_dim, num_layers = 1, batch_first = TRUE)
        self$policy_head <- nn_linear(hidden_dim, 1)
        self$pi_head <- nn_linear(hidden_dim, K)
        self$mu_head <- nn_linear(hidden_dim, K)
        self$sigma_head <- nn_linear(hidden_dim, K)
        self$tau_head <- nn_linear(hidden_dim, K)
        self$log_var_policy <- nn_parameter(torch_zeros(1))
        self$log_var_kin <- nn_parameter(torch_zeros(1))
        self$K <- K
    },
    forward = function(x, h0 = NULL) {
        out <- self$gru(x, h0)
        h <- out[[1]]
        logits_p <- torch_clamp(self$policy_head(h), min = -15, max = 15)
        p_switch <- nnf_sigmoid(logits_p)
        pi_mix <- nnf_softmax(torch_clamp(self$pi_head(h), min = -15, max = 15), dim = -1)
        mu_rt <- self$mu_head(h)
        sigma_rt <- nnf_softplus(torch_clamp(self$sigma_head(h), min = -15, max = 15)) + 1e-4
        tau_rt <- nnf_sigmoid(torch_clamp(self$tau_head(h), min = -15, max = 15)) * (0.99 * min_RT)
        list(p = p_switch, pi = pi_mix, mu = mu_rt, sigma = sigma_rt, tau = tau_rt)
    }
)

cat("Loading frozen model...\n")
model <- torch_load("../../results/frozen_rnn_baseline.pt", device = "cpu")
model <- model$to(device = device)
model$eval()

all_probs <- numeric()
all_labels <- numeric()
all_subjs <- numeric()

cat("Computing predictions...\n")
with_no_grad({
    for (s in rnn_phase1_subjs) {
        idx <- which(stan_data$subj == s)
        x_t <- torch_tensor(X[idx, ], dtype=torch_float(), device=device)$unsqueeze(1)
        
        preds <- model(x_t)
        
        # mask out the first trial of each block/subject where switch is undefined (0)
        mask <- c(FALSE, rep(TRUE, length(idx)-1))
        
        p_t <- as.numeric(preds$p$cpu())
        y_t <- Switch[idx]
        
        all_probs <- c(all_probs, p_t[mask])
        all_labels <- c(all_labels, y_t[mask])
        all_subjs <- c(all_subjs, rep(s, sum(mask)))
    }
})

results_df <- data.frame(
    subj = all_subjs,
    pred_prob = all_probs,
    true_label = all_labels
)

# Compute quick ECE (10 bins) for logging
breaks <- seq(0, 1, by = 0.1)
results_df$bin <- cut(results_df$pred_prob, breaks = breaks, include.lowest = TRUE, labels = FALSE)

ece_summary <- results_df %>%
    group_by(bin) %>%
    summarize(
        count = n(),
        conf = mean(pred_prob),
        acc = mean(true_label)
    ) %>%
    mutate(
        weight = count / sum(count),
        err = abs(conf - acc)
    )

ece <- sum(ece_summary$weight * ece_summary$err)
cat(sprintf("Calculated ECE (10 bins): %.4f\n", ece))

saveRDS(results_df, "../../results/ECE.rds")
cat("Saved to ../../results/ECE.rds\n")
