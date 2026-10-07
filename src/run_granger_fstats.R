
library(lmtest)
library(dplyr)

targets <- readRDS("src/r/distillation_targets_v2.rds")
N_trials <- length(targets$labels)
unique_subjs <- unique(targets$subjs)

EMA_Left <- numeric(N_trials); Reward_Ent <- numeric(N_trials); EMA_BinState <- numeric(N_trials)
BinaryState <- ifelse(targets$X[,1] > targets$X[,2], 1, 0)
window_size <- 15; alpha_ema <- 2 / (5 + 1); current_subj <- -1
for (t in 1:N_trials) {
  if (targets$subjs[t] != current_subj) { current_ema_left <- 0.5; current_ema_bin <- 0.5; current_subj <- targets$subjs[t] }
  left_outcome <- ifelse(targets$lag_resp[t] == 1, targets$lag_reward[t], 1 - targets$lag_reward[t])
  current_ema_left <- alpha_ema * left_outcome + (1 - alpha_ema) * current_ema_left; EMA_Left[t] <- current_ema_left
  current_ema_bin <- alpha_ema * BinaryState[t] + (1 - alpha_ema) * current_ema_bin; EMA_BinState[t] <- current_ema_bin
}
for(s in unique_subjs) {
    idx <- which(targets$subjs == s); subj_rew <- targets$lag_reward[idx]
    for(t in 1:length(idx)) {
        if(t == 1) { p <- 0.5 } else if (t <= window_size) { p <- mean(subj_rew[1:t]) } else { p <- mean(subj_rew[(t - window_size + 1):t]) }
        p <- max(1e-6, min(1 - 1e-6, p)); Reward_Ent[idx[t]] <- -p*log2(p) - (1-p)*log2(1-p)
    }
}

Y_Val <- scale(EMA_Left)
Y_Vol <- scale(Reward_Ent)
Y_State <- scale(EMA_BinState)

# Load the intact mu and G/Z/W/D if needed. 
# But wait, earlier I just used a simpler proxy for Cb vs Ctx.
# Actually, the user already had Granger results. Let me check how I ran Granger previously.

