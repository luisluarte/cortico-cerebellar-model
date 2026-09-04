library(torch)
library(mco)

df <- read.csv('data/behavioral_compilate.csv')
df <- df[df$Resp %in% c(1, 2), ]
subjs <- readRDS('excluded_subjects.rds')
df_50 <- df[df$participant_id %in% subjs, ]

L_pure <- readRDS('L_pure.rds')

# Extract Fold 1 for Evolutionary Optimization
K <- 5
set.seed(42)
folds <- sample(rep(1:K, length.out = length(subjs)))
test_subjs <- subjs[folds == 1]
train_subjs <- subjs[folds != 1]
baseline_L_pure <- mean(L_pure[test_subjs])

HybridRNN <- nn_module(
  "HybridRNN",
  initialize = function(input_dim = 9, hidden_dim = 20, output_dim = 8, N_ratio = 5, p = 0.1) {
    self$rnn <- nn_rnn(input_dim, hidden_dim, batch_first = TRUE)
    self$fc_rnn <- nn_linear(hidden_dim, output_dim)
    
    N_granule <- as.integer(round(hidden_dim * N_ratio))
    self$fc_pc <- nn_linear(N_granule, output_dim)
    
    # Generate Achlioptas matrix
    elements <- sample(c(-sqrt(3), 0, sqrt(3)), N_granule * hidden_dim, replace=TRUE, prob=c(p/2, 1-p, p/2))
    W_mat <- matrix(elements, nrow=hidden_dim, ncol=N_granule) # transposed for matmul: [d, N]
    self$W_res <- torch_tensor(W_mat, dtype=torch_float(), requires_grad=FALSE)
  },
  forward = function(x, bd1, bd2) {
    out <- self$rnn(x)
    h_t <- out[[1]] # [1, seq_len, d]
    
    # A2: Sever gradients to granular expansion
    h_t_detached <- h_t$detach()
    z_t <- torch_matmul(h_t_detached, self$W_res) # [1, seq_len, N]
    
    logits_rnn <- self$fc_rnn(h_t)
    logits_pc <- self$fc_pc(z_t)
    
    logits <- logits_rnn + logits_pc
    
    idx_bd1 <- bd1$unsqueeze(2)$to(torch_int64())
    idx_bd2 <- bd2$unsqueeze(2)$to(torch_int64())
    logits_sq <- logits$squeeze(1)
    
    val1 <- logits_sq$gather(2, idx_bd1)
    val2 <- logits_sq$gather(2, idx_bd2)
    
    return(torch_cat(list(val1, val2), dim = 2))
  }
)

evaluate_genome <- function(params) {
  N_ratio <- params[1]
  p <- params[2]
  
  model <- HybridRNN(9, 20, 8, N_ratio, p)
  optimizer <- optim_adam(model$parameters, lr = 0.01)
  
  # Train
  for (epoch in 1:5) { 
    for (s in train_subjs) {
      dat <- df_50[df_50$participant_id == s, ]
      seq_len <- nrow(dat)
      x <- matrix(0, nrow = seq_len, ncol = 9)
      if (seq_len > 1) {
        prev_chosen <- ifelse(dat$Resp[-seq_len] == 1, dat$Bd1[-seq_len], dat$Bd2[-seq_len])
        for (t in 2:seq_len) {
          x[t, prev_chosen[t-1]] <- 1 
          x[t, 9] <- dat$F[t-1]      
        }
      }
      x_t <- torch_tensor(x, dtype = torch_float())$unsqueeze(1)
      bd1_t <- torch_tensor(dat$Bd1, dtype = torch_long())
      bd2_t <- torch_tensor(dat$Bd2, dtype = torch_long())
      resp_t <- torch_tensor(dat$Resp, dtype = torch_long())
      
      optimizer$zero_grad()
      preds <- model(x_t, bd1_t, bd2_t)
      loss <- nnf_cross_entropy(preds, resp_t)
      loss$backward()
      optimizer$step()
    }
  }
  
  # Eval
  L_hybrid_vec <- numeric(length(test_subjs))
  with_no_grad({
    for (i in seq_along(test_subjs)) {
      s <- test_subjs[i]
      dat <- df_50[df_50$participant_id == s, ]
      seq_len <- nrow(dat)
      x <- matrix(0, nrow = seq_len, ncol = 9)
      if (seq_len > 1) {
        prev_chosen <- ifelse(dat$Resp[-seq_len] == 1, dat$Bd1[-seq_len], dat$Bd2[-seq_len])
        for (t in 2:seq_len) {
          x[t, prev_chosen[t-1]] <- 1 
          x[t, 9] <- dat$F[t-1]      
        }
      }
      x_t <- torch_tensor(x, dtype = torch_float())$unsqueeze(1)
      bd1_t <- torch_tensor(dat$Bd1, dtype = torch_long())
      bd2_t <- torch_tensor(dat$Bd2, dtype = torch_long())
      resp_t <- torch_tensor(dat$Resp, dtype = torch_long())
      
      preds <- model(x_t, bd1_t, bd2_t)
      loss <- nnf_cross_entropy(preds, resp_t)
      L_hybrid_vec[i] <- loss$item()
    }
  })
  
  L_hybrid <- mean(L_hybrid_vec)
  
  # Objective 1: Kinematic Parity
  delta_L <- abs(L_hybrid - baseline_L_pure)
  
  # Objective 2: Pathway Reliance (Offloading)
  w_rnn <- as.numeric(model$fc_rnn$weight)
  w_pc <- as.numeric(model$fc_pc$weight)
  
  norm_rnn <- sum(w_rnn^2)
  norm_pc <- sum(w_pc^2)
  
  reliance <- norm_pc / (norm_pc + norm_rnn)
  
  # nsga2 minimizes all objectives. We want to maximize reliance, so we minimize -reliance
  cat(sprintf("[Eval] N=%.1f, p=%.3f -> Delta L=%.4f, Offloading=%.2f%%\\n", N_ratio, p, delta_L, reliance * 100))
  return(c(delta_L, -reliance))
}

cat("Starting NSGA-II Evolution (Pop: 20, Gen: 10)...\\n")
res <- nsga2(evaluate_genome, idim = 2, odim = 2, 
             lower.bounds = c(1, 0.01), upper.bounds = c(20, 0.5), 
             popsize = 20, generations = 10)

saveRDS(res, 'nsga2_results.rds')
cat("Evolution complete! Saved to nsga2_results.rds\\n")
