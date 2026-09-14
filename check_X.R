targets <- readRDS('src/r/distillation_targets_v2.rds')
Bd1_val <- apply(targets$X[, 1:8], 1, function(r) { w <- which(r==1); if(length(w)>0) w[1] else NA })
Bd1_scaled <- (Bd1_val - min(Bd1_val, na.rm=T)) / (max(Bd1_val, na.rm=T) - min(Bd1_val, na.rm=T))

Bd2_val <- apply(targets$X[, 9:16], 1, function(r) { w <- which(r==1); if(length(w)>0) w[1] else NA })
Bd2_scaled <- (Bd2_val - min(Bd2_val, na.rm=T)) / (max(Bd2_val, na.rm=T) - min(Bd2_val, na.rm=T))

lag_Reward <- targets$lag_reward
lag_Resp <- targets$lag_resp

lag_Ch_val <- apply(targets$X[, 19:26], 1, function(r) { w <- which(r==1); if(length(w)>0) w[1] else NA })
lag_Ch_scaled <- (lag_Ch_val - min(lag_Ch_val, na.rm=T)) / (max(lag_Ch_val, na.rm=T) - min(lag_Ch_val, na.rm=T))

ITI_scaled <- targets$ITI / max(targets$ITI, na.rm=T)

X_6 <- cbind(Bd1_scaled, Bd2_scaled, lag_Reward, lag_Ch_scaled, lag_Resp, ITI_scaled)

d <- readRDS('data/dataset_A.rds')
d_X <- as.matrix(d[, c("Bd1_scaled", "Bd2_scaled", "lag_Reward", "lag_Ch", "lag_Resp", "ITI_scaled")])

cat("X_6 head:\n")
print(head(X_6))
cat("\nd_X head:\n")
print(head(d_X))
