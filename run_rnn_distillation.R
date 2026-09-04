library(torch)
library(dplyr)
library(jsonlite)

stan_data <- read_json('data/stan_data_N30.json', simplifyVector = TRUE)
N_trials <- stan_data$N
N_subj <- stan_data$N_subj

X_features <- matrix(0, nrow = N_trials, ncol = 8)
for (i in 1:N_trials) {
  X_features[i, stan_data$Bd1[i]] <- 1
  X_features[i, stan_data$Bd2[i]] <- 1
}

lag_Reward <- c(0, stan_data$Reward[-N_trials])
lag_Resp <- c(0, stan_data$Resp[-N_trials] - 1)
first_trials <- c(1, which(diff(stan_data$subj) != 0) + 1)
lag_Reward[first_trials] <- 0
lag_Resp[first_trials] <- 0

X <- cbind(X_features, lag_Reward, lag_Resp)
input_dim <- ncol(X)
hidden_dim <- 32

UniversalRNN <- nn_module(
  "UniversalRNN",
  initialize = function(input_dim, hidden_dim) {
    self$gru <- nn_gru(input_dim, hidden_dim, batch_first = TRUE)
    self$policy_head <- nn_linear(hidden_dim, 1)
    self$mu_head <- nn_linear(hidden_dim, 1)
    self$sigma_head <- nn_linear(hidden_dim, 1)
  },
  forward = function(x, h0 = NULL) {
    out <- self$gru(x, h0)
    h <- out[[1]]
    p_choice <- nnf_sigmoid(self$policy_head(h))
    mu_rt <- self$mu_head(h)
    sigma_rt <- nnf_softplus(self$sigma_head(h)) + 1e-4
    list(p = p_choice, mu = mu_rt, sigma = sigma_rt, h = h)
  }
)

model <- UniversalRNN(input_dim, hidden_dim)
optimizer <- optim_adam(model$parameters, lr = 0.01)

epochs <- 50
for (ep in 1:epochs) {
  loss_total <- 0
  for (s in 1:N_subj) {
    idx <- which(stan_data$subj == s)
    x_t <- torch_tensor(X[idx, ], dtype = torch_float())$unsqueeze(1)
    y_choice <- torch_tensor(stan_data$Resp[idx] - 1, dtype = torch_float())$unsqueeze(1)$unsqueeze(3)
    y_rt <- torch_tensor(stan_data$RT[idx], dtype = torch_float())$unsqueeze(1)$unsqueeze(3)
    
    optimizer$zero_grad()
    preds <- model(x_t)
    loss_policy <- nnf_binary_cross_entropy(preds$p, y_choice)
    log_rt <- torch_log(y_rt)
    dist <- distr_normal(preds$mu, preds$sigma)
    loss_rt <- -dist$log_prob(log_rt)$mean()
    loss <- loss_policy + loss_rt
    loss$backward()
    optimizer$step()
    loss_total <- loss_total + loss$item()
  }
  if (ep %% 10 == 0) cat(sprintf("Epoch %d | Loss: %.4f\n", ep, loss_total / N_subj))
}

model$eval()
soft_p <- numeric()
soft_mu <- numeric()
soft_sigma <- numeric()
H <- NULL

for (s in 1:N_subj) {
  idx <- which(stan_data$subj == s)
  x_t <- torch_tensor(X[idx, ], dtype = torch_float())$unsqueeze(1)
  with_no_grad({ preds <- model(x_t) })
  soft_p <- c(soft_p, as.numeric(preds$p))
  soft_mu <- c(soft_mu, as.numeric(preds$mu))
  soft_sigma <- c(soft_sigma, as.numeric(preds$sigma))
  if (s == 1) { H <- as.matrix(preds$h[1, , ]) } else { H <- rbind(H, as.matrix(preds$h[1, , ])) }
}

stan_data$soft_p <- soft_p
stan_data$soft_mu <- soft_mu
stan_data$soft_sigma <- soft_sigma
stan_data$H <- H
stan_data$hidden_dim <- hidden_dim

write_json(stan_data, 'data/stan_data_N30_distilled.json', auto_unbox = TRUE)
cat("Distillation complete.\n")
