library(torch)
library(dplyr)

df <- read.csv('data/behavioral_compilate.csv')
# Filter out timeouts (Resp == 3)
df <- df[df$Resp %in% c(1, 2), ]
subjs <- unique(df$participant_id)[1:50]
saveRDS(subjs, 'excluded_subjects.rds')
df_50 <- df[df$participant_id %in% subjs, ]

PureRNN <- nn_module(
  "PureRNN",
  initialize = function(input_dim = 9, hidden_dim = 20, output_dim = 8) {
    self$rnn <- nn_rnn(input_dim, hidden_dim, batch_first = TRUE)
    self$fc <- nn_linear(hidden_dim, output_dim)
  },
  forward = function(x, bd1, bd2) {
    out <- self$rnn(x)
    logits <- self$fc(out[[1]])
    
    seq_len <- x$shape[2]
    idx_bd1 <- bd1$unsqueeze(2)$to(torch_int64())
    idx_bd2 <- bd2$unsqueeze(2)$to(torch_int64())
    
    logits_sq <- logits$squeeze(1)
    
    val1 <- logits_sq$gather(2, idx_bd1)
    val2 <- logits_sq$gather(2, idx_bd2)
    
    presented_logits <- torch_cat(list(val1, val2), dim = 2)
    return(presented_logits)
  }
)

K <- 5
set.seed(42)
folds <- sample(rep(1:K, length.out = length(subjs)))

L_pure <- numeric(length(subjs))
names(L_pure) <- subjs

for (k in 1:K) {
  cat(sprintf("Fold %d/%d\\n", k, K))
  test_subjs <- subjs[folds == k]
  train_subjs <- subjs[folds != k]
  
  model <- PureRNN(9, 20, 8)
  optimizer <- optim_adam(model$parameters, lr = 0.01)
  
  for (epoch in 1:5) { 
    epoch_loss <- 0
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
  
  with_no_grad({
    for (s in test_subjs) {
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
      L_pure[s] <- loss$item()
    }
  })
}

saveRDS(L_pure, 'L_pure.rds')
cat("Saved L_pure.rds\\n")
cat("Mean L_pure:", mean(L_pure), "\\n")
