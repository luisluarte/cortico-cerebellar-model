
library(torch)
library(dplyr)

cat("Starting 02_rnn_oracle.R...\n")
data_A <- readRDS("../../data/dataset_A.rds")
data_D <- readRDS("../../data/dataset_D.rds")

prepare_tensors <- function(d) {
  subjs <- unique(d$participant_id)
  max_len <- max(table(d$participant_id))
  
  X_array <- array(0, dim=c(length(subjs), max_len, 5))
  Y_array <- array(0, dim=c(length(subjs), max_len, 1))
  mask_array <- array(0, dim=c(length(subjs), max_len))
  
  for(i in 1:length(subjs)) {
    s_d <- d[d$participant_id == subjs[i], ]
    n <- nrow(s_d)
    X_array[i, 1:n, 1] <- s_d$Bd1_scaled
    X_array[i, 1:n, 2] <- s_d$Bd2_scaled
    X_array[i, 1:n, 3] <- s_d$lag_Reward
    X_array[i, 1:n, 4] <- s_d$lag_Resp
    X_array[i, 1:n, 5] <- s_d$ITI_scaled
    Y_array[i, 1:n, 1] <- s_d$Resp_mapped
    mask_array[i, 1:n] <- 1
  }
  
  return(list(
    X = torch_tensor(X_array, dtype=torch_float32()),
    Y = torch_tensor(Y_array, dtype=torch_float32()),
    mask = torch_tensor(mask_array, dtype=torch_float32()),
    subjs = subjs
  ))
}

train_rnn <- function(X, Y, mask, hidden_size, lr, epochs=50) {
  model <- nn_module(
    "RNNModel",
    initialize = function(input_size, hidden_size) {
      self$rnn <- nn_gru(input_size, hidden_size, batch_first = TRUE)
      self$fc <- nn_linear(hidden_size, 1)
    },
    forward = function(x) {
      out <- self$rnn(x)
      logits <- self$fc(out[[1]])
      return(list(logits = logits, hidden = out[[1]]))
    }
  )
  
  net <- model(5, hidden_size)
  opt <- optim_adam(net$parameters, lr=lr)
  loss_fn <- nn_bce_with_logits_loss(reduction="none")
  
  for(e in 1:epochs) {
    opt$zero_grad()
    preds <- net(X)
    loss_mat <- loss_fn(preds$logits, Y)
    loss <- (loss_mat$squeeze() * mask)$sum() / mask$sum()
    loss$backward()
    opt$step()
  }
  return(net)
}

# 1. Grid Search on Dataset A
cat("Running grid search on Dataset A...\n")
tensors_A <- prepare_tensors(data_A)
grid <- expand.grid(hidden_size = c(16, 32, 64), lr = c(0.01, 0.005))
best_loss <- Inf
best_params <- NULL

for(i in 1:nrow(grid)) {
  hs <- grid$hidden_size[i]
  lr <- grid$lr[i]
  net <- train_rnn(tensors_A$X, tensors_A$Y, tensors_A$mask, hs, lr, epochs=40)
  
  with_no_grad({
    preds <- net(tensors_A$X)
    loss_mat <- nnf_binary_cross_entropy_with_logits(preds$logits, tensors_A$Y, reduction="none")
    loss <- as.numeric((loss_mat$squeeze() * tensors_A$mask)$sum() / tensors_A$mask$sum())
  })
  
  cat(sprintf("HS: %d | LR: %f | Loss: %f\n", hs, lr, loss))
  if(loss < best_loss) {
    best_loss <- loss
    best_params <- grid[i,]
  }
}

cat("Best parameters:\n")
print(best_params)

# 2. LSNO (10 iterations) on Dataset D
cat("Running LSNO on Dataset D...\n")
tensors_D <- prepare_tensors(data_D)
subjs_D <- tensors_D$subjs

folds <- 10
fold_size <- floor(length(subjs_D) / folds)
lsno_results <- numeric(folds)

hs <- best_params$hidden_size
lr <- best_params$lr

for(k in 1:folds) {
  test_idx <- ((k-1)*fold_size + 1):(k*fold_size)
  if(k == folds) test_idx <- ((k-1)*fold_size + 1):length(subjs_D)
  
  train_idx <- setdiff(1:length(subjs_D), test_idx)
  
  X_tr <- tensors_D$X[train_idx, , ]
  Y_tr <- tensors_D$Y[train_idx, , ]
  mask_tr <- tensors_D$mask[train_idx, ]
  
  X_te <- tensors_D$X[test_idx, , ]
  Y_te <- tensors_D$Y[test_idx, , ]
  mask_te <- tensors_D$mask[test_idx, ]
  
  net_k <- train_rnn(X_tr, Y_tr, mask_tr, hs, lr, epochs=50)
  
  with_no_grad({
    preds_te <- net_k(X_te)
    loss_mat <- nnf_binary_cross_entropy_with_logits(preds_te$logits, Y_te, reduction="none")
    loss_te <- as.numeric((loss_mat$squeeze() * mask_te)$sum() / mask_te$sum())
  })
  lsno_results[k] <- loss_te
  cat(sprintf("Fold %d Test Loss: %f\n", k, loss_te))
}

saveRDS(lsno_results, "../../results/lsno_results.rds")

# 3. Full Oracle Training on Dataset D
cat("Training Oracle on complete Dataset D...\n")
oracle_net <- train_rnn(tensors_D$X, tensors_D$Y, tensors_D$mask, hs, lr, epochs=80)

torch_save(oracle_net, "../../results/oracle_weights.pt")

with_no_grad({
  out <- oracle_net(tensors_D$X)
  logits <- as_array(out$logits)
  hidden <- as_array(out$hidden)
})

targets_list <- list()
for(i in 1:length(subjs_D)) {
  s_d <- data_D[data_D$participant_id == subjs_D[i], ]
  n <- nrow(s_d)
  targets_list[[subjs_D[i]]] <- list(
    logits = logits[i, 1:n, 1],
    hidden = hidden[i, 1:n, ]
  )
}

saveRDS(targets_list, "../../results/oracle_targets.rds")
cat("Step 2 Complete! Targets saved.\n")
