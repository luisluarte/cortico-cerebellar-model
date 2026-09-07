
library(torch)
SpatialRNN <- nn_module("SpatialRNN",
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
        tau_rt <- nnf_sigmoid(torch_clamp(self$tau_head(h), min = -15, max = 15)) * (0.99 * 0.1)
        list(p = p_switch, pi = pi_mix, mu = mu_rt, sigma = sigma_rt, tau = tau_rt)
    }
)
model <- torch_load("../../results/frozen_rnn_baseline.pt", device="cpu")
model$eval()
x_t <- torch_randn(c(100, 1, 36))
with_no_grad({
    cat("Predicting...
")
    preds <- model(x_t)
    cat("p shape:", dim(preds$p), "
")
    p <- as_array(preds$p)
    cat("Extracted p
")

    sigma <- as_array(preds$sigma$cpu())
    cat("Extraction worked
")
})
