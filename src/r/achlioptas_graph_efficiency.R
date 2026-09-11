library(ggplot2)
library(dplyr)
library(gridExtra)
library(grid)
library(doParallel)
library(foreach)

N_G_list <- c(32, 64, 128, 256, 512, 1024)
P_seq <- seq(10, 450, by=10)
B_bootstraps <- 3
pts_per_manifold <- 15
lambda_ridge <- 1.0
error_threshold <- 0.02

# Achlioptas Matrix (90% zeros, 5% +1, 5% -1)
create_achlioptas <- function(n_in, n_out) {
  W <- matrix(0, nrow=n_in, ncol=n_out)
  vals <- c(-1, 0, 1)
  probs <- c(0.05, 0.90, 0.05)
  W[] <- sample(vals, n_in * n_out, replace=TRUE, prob=probs)
  return(W)
}

# Setup Parallel Backend (Windows Compatible PSOCK Cluster)
num_cores <- parallel::detectCores() - 1
cat(sprintf("Firing up %d CPU cores for maximum parallel sweep...\n", num_cores))
cl <- makeCluster(num_cores)
registerDoParallel(cl)

results_err <- foreach(i = 1:length(P_seq), .combine=rbind) %dopar% {
  P <- P_seq[i]
  err_temp <- matrix(0, nrow=B_bootstraps, ncol=length(N_G_list))
  
  for (b in 1:B_bootstraps) {
    # Ensure independent random streams per parallel worker
    set.seed(42 + i * 100 + b)
    
    bases <- matrix(runif(P * 6, -1, 1), nrow=P, ncol=6)
    velocities <- matrix(runif(P * 6, -1, 1), nrow=P, ncol=6)
    manifold_labels <- sample(c(-1, 1), P, replace = TRUE)
    
    X <- matrix(0, nrow = P * pts_per_manifold, ncol = 6)
    Y_labels <- numeric(P * pts_per_manifold)
    idx <- 1
    t_steps <- seq(-1, 1, length.out = pts_per_manifold)
    for (m in 1:P) {
      for (t in t_steps) {
        X[idx, ] <- bases[m, ] + t * velocities[m, ] + rnorm(6, 0, 0.05)
        Y_labels[idx] <- manifold_labels[m]
        idx <- idx + 1
      }
    }
    
    # Linear Cortical Projection (No ReLU!)
    W_in <- matrix(rnorm(6 * 32, mean=0, sd=1/sqrt(6)), nrow=6, ncol=32)
    Y_C <- X %*% W_in
    
    for (j in seq_along(N_G_list)) {
      N_G <- N_G_list[j]
      
      # Achlioptas Linear Projection (90% sparse)
      W_GC <- create_achlioptas(32, N_G)
      Y_G <- Y_C %*% W_GC
      
      # THE TRUE NON-LINEARITY: Quadratic Expansion (G * G)
      # This simulates the element-wise eligibility trace modulation (G % W_purk)
      Y_Q <- Y_G * Y_G 
      
      # SVD WHITENING on the Quadratic state
      Y_Q_centered <- scale(Y_Q, scale = FALSE)
      svd_res <- svd(Y_Q_centered)
      keep <- svd_res$d > 1e-5
      
      if(sum(keep) == 0) {
        err_temp[b, j] <- 0.5
        next
      }
      
      Y_white <- svd_res$u[, keep, drop=FALSE] * sqrt(nrow(Y_Q_centered) - 1)
      
      # Ridge Regression Readout
      Y_W_T_Y_W <- crossprod(Y_white)
      diag(Y_W_T_Y_W) <- diag(Y_W_T_Y_W) + lambda_ridge
      
      W_out <- solve(Y_W_T_Y_W, crossprod(Y_white, Y_labels))
      err_temp[b, j] <- mean(sign(Y_white %*% W_out) != Y_labels)
    }
  }
  err_temp
}
stopImplicitCluster()

# Aggregate parallel results
agg_err <- matrix(0, nrow=length(P_seq), ncol=length(N_G_list))
for(i in 1:length(P_seq)) {
  idx <- ((i-1)*B_bootstraps + 1):(i*B_bootstraps)
  agg_err[i, ] <- colMeans(results_err[idx, , drop=FALSE])
}

P_max_list <- numeric(length(N_G_list))
for (j in seq_along(N_G_list)) {
  valid_P <- P_seq[agg_err[, j] <= error_threshold]
  if (length(valid_P) > 0) {
    P_max_list[j] <- max(valid_P)
  } else {
    P_max_list[j] <- 0
  }
}

df_capacity <- data.frame(N_G = N_G_list, P_max = P_max_list)
Wiring_Cost <- N_G_list^(4/3)
Efficiency <- P_max_list / Wiring_Cost

df <- data.frame(N_G=N_G_list, P_max=P_max_list, Wiring_Cost, Efficiency)
df$Norm_P_max <- df$P_max / max(df$P_max)
df$Norm_Cost <- df$Wiring_Cost / max(df$Wiring_Cost)

# Plot 1: Benefit vs Cost
p1 <- ggplot(df, aes(x=N_G)) +
  geom_line(aes(y=Norm_P_max, color="Quadratic Memory Capacity"), linewidth=1.5) +
  geom_point(aes(y=Norm_P_max, color="Quadratic Memory Capacity"), size=3) +
  geom_line(aes(y=Norm_Cost, color="3D Wiring Volume (N^1.33)"), linewidth=1.5, linetype="dashed") +
  geom_point(aes(y=Norm_Cost, color="3D Wiring Volume (N^1.33)"), size=3) +
  scale_color_manual(values=c("Quadratic Memory Capacity"="#2980b9", "3D Wiring Volume (N^1.33)"="#c0392b")) +
  scale_x_continuous(breaks=N_G_list) +
  theme_minimal(base_size=14) +
  theme(legend.position="bottom", legend.title=element_blank(), plot.title = element_text(face="bold")) +
  labs(title="1. Achlioptas Quadratic Tradeoff", x="Granule Size", y="Normalized Scale")

peak_NG <- df$N_G[which.max(df$Efficiency)]

p2 <- ggplot(df, aes(x=N_G, y=Efficiency)) +
  geom_line(color="#27ae60", linewidth=1.5) +
  geom_point(size=5, color="#2ecc71") +
  geom_vline(xintercept=peak_NG, linetype="dotted", color="black", linewidth=1) +
  annotate("text", x=peak_NG+40, y=max(df$Efficiency)*0.9, label=sprintf("Optimal Size: %d", peak_NG), fontface="bold") +
  scale_x_continuous(breaks=N_G_list) +
  theme_minimal(base_size=14) +
  theme(plot.title = element_text(face="bold")) +
  labs(title="2. Whitened Cost-Efficiency", subtitle="Capacity / 3D Wiring Volume", x="Granule Size", y="Efficiency")

p_final <- grid.arrange(p1, p2, ncol=1, top=grid::textGrob("Graph Optimization: Achlioptas Quadratic Network", gp=grid::gpar(fontsize=18, fontface="bold")))

ggsave("results/achlioptas_graph_efficiency.png", p_final, width=10, height=9, bg="white")
write.csv(df, "results/achlioptas_efficiency.csv", row.names=F)
cat("Parallel Achlioptas Whitened Test Complete!\n")
