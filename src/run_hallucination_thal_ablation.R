
library(Rcpp)
library(RcppArmadillo)
library(ggplot2)
library(dplyr)

cat("Loading Simulation Core and Data...\n")
sourceCpp("src/r/sim_bio_probes.cpp")
sourceCpp("src/r/sim_hallucination.cpp")

targets <- readRDS("src/r/distillation_targets_v2.rds")
X_emp <- targets$X
input_dim <- ncol(X_emp)
max_cells <- 896

res <- readRDS("results/nsga2_pruning_results.rds")
opt_p <- res$par[1, ]

n_trials <- 10
syn_ITI <- rep(0.1, n_trials)
syn_Pi <- matrix(1, nrow=n_trials, ncol=input_dim)
syn_X <- matrix(0, nrow=n_trials, ncol=input_dim)
syn_X[, 1] <- 1.0 
syn_X[, 2] <- 0.0 
syn_X[, 3] <- 1.0 
syn_X[, 4] <- 1.0 
syn_X[, 5] <- 1.0 
syn_X[, 6] <- 0.5 

dt <- 0.1
n_steps <- 200
N_sims <- 100

df_all <- data.frame()
set.seed(42)

calc_ent <- function(mat) {
    apply(mat, 1, function(v) {
        v <- abs(v)
        s <- sum(v)
        if(s < 1e-12) return(0)
        p <- v / s
        p <- p[p > 0]
        -sum(p * log2(p))
    })
}

normalize_01 <- function(x) {
    mn <- min(x, na.rm=TRUE)
    mx <- max(x, na.rm=TRUE)
    if(abs(mx - mn) < 1e-12) return(rep(0, length(x)))
    return((x - mn) / (mx - mn))
}

for(s in 1:N_sims) {
    if(s %% 10 == 0) cat(sprintf("Sim %d / %d...\n", s, N_sims))
    
    W_gen <- matrix(rnorm(input_dim * 32, 0, 1/sqrt(32)), nrow=input_dim, ncol=32)
    W_ach1 <- matrix(rnorm(max_cells * 32, 0, 1/sqrt(32)), nrow=max_cells, ncol=32)
    mask1 <- matrix(rbinom(max_cells * 32, 1, 0.1), nrow=max_cells, ncol=32)
    W_ach1 <- W_ach1 * mask1
    W_ach2 <- matrix(rnorm(max_cells * max_cells, 0, 1/sqrt(max_cells)), nrow=max_cells, ncol=max_cells)
    W_thal <- matrix(rnorm(32 * max_cells, 0, 1/sqrt(max_cells)), nrow=32, ncol=max_cells)
    
    # 1. Baseline
    hal_base <- simulate_hallucination(n_trials - 1, n_steps, dt, syn_X, syn_ITI, W_gen, W_ach1, W_ach2, W_thal, syn_Pi,
                                  opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], opt_p[8], opt_p[9])

    # 2. Ablated Thalamic Feedback (beta_thal = 0)
    hal_abl <- simulate_hallucination(n_trials - 1, n_steps, dt, syn_X, syn_ITI, W_gen, W_ach1, W_ach2, W_thal, syn_Pi,
                                  opt_p[1], opt_p[2], 0.0, opt_p[4], opt_p[5], opt_p[6], opt_p[7], opt_p[8], opt_p[9])

    Time_seq <- seq(dt, n_steps * dt, by=dt)
    
    df_all <- rbind(df_all, data.frame(
        Sim = s,
        Time = rep(Time_seq, 6),
        Layer = c(rep("Cortical (mu)", n_steps), rep("DCN (D)", n_steps), rep("Granule (P)", n_steps),
                  rep("Cortical (mu)", n_steps), rep("DCN (D)", n_steps), rep("Granule (P)", n_steps)),
        Condition = c(rep("Baseline", n_steps*3), rep("Thalamic Ablation", n_steps*3)),
        Entropy = c(normalize_01(calc_ent(hal_base$mu_hallucinated)),
                    normalize_01(calc_ent(hal_base$D_hallucinated)),
                    normalize_01(calc_ent(hal_base$P_hallucinated)),
                    normalize_01(calc_ent(hal_abl$mu_hallucinated)),
                    normalize_01(calc_ent(hal_abl$D_hallucinated)),
                    normalize_01(calc_ent(hal_abl$P_hallucinated)))
    ))
}

df_summary <- df_all %>%
    group_by(Time, Layer, Condition) %>%
    summarize(
        Mean = mean(Entropy),
        SE = sd(Entropy) / sqrt(n()),
        .groups = "drop"
    )

# Plot Layer Comparisons
plot_ablation <- function(layer_name, fname) {
    df_plot <- df_summary %>% filter(Layer == layer_name)
    
    p <- ggplot(df_plot, aes(x=Time, y=Mean, color=Condition, fill=Condition)) +
        geom_ribbon(aes(ymin = Mean - SE, ymax = Mean + SE), alpha=0.2, color=NA) +
        geom_line(linewidth=1.2) +
        scale_color_manual(values=c("Baseline"="#e41a1c", "Thalamic Ablation"="#377eb8")) +
        scale_fill_manual(values=c("Baseline"="#e41a1c", "Thalamic Ablation"="#377eb8")) +
        labs(title=paste("Thalamic Feedback Ablation:", layer_name), 
             subtitle="Normalized Entropy (Mean ± 1 SE, 100 Sims)", 
             y="Normalized Entropy [0, 1]", x="Rest Time (s)") +
        scale_x_continuous(breaks = seq(0, 20, by = 5)) +
        theme_minimal(base_size = 14) +
        theme(legend.position="bottom", legend.title=element_blank())
        
    ggsave(fname, plot=p, width=7, height=5, dpi=300)
}

plot_ablation("Cortical (mu)", "results/Fig13_Ablation_Cortical.png")
plot_ablation("DCN (D)", "results/Fig13_Ablation_DCN.png")
plot_ablation("Granule (P)", "results/Fig13_Ablation_Granule.png")

cat("Done!\n")

