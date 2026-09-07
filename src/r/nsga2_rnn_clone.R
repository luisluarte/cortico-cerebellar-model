local_lib <- Sys.getenv("R_LIBS_USER")
.libPaths(c(local_lib, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"))

suppressPackageStartupMessages({
    library(mco)
})

cat("Loading Distillation Targets...\n")
targets <- readRDS("distillation_targets_v2.rds")

# Calibrate the frozen teacher logits (Platt Scaling)
cat("Calibrating the Teacher's Logits via Platt Scaling...\n")
fit <- glm(targets$labels ~ targets$rnn_logits, family = binomial)
cat(sprintf("Platt Scaling - Intercept: %.4f | Slope: %.4f\n", coef(fit)[1], coef(fit)[2]))

calibrated_rnn_logits <- predict(fit, type = "link")
teacher_soft_targets <- plogis(calibrated_rnn_logits)
p_val <- teacher_soft_targets

# The "Biological" model in this sanity check is an exact clone of the RNN
bio_logits <- calibrated_rnn_logits
bio_mu <- targets$rnn_mu


# The exact RNN Teacher Outputs
rnn_mu <- targets$rnn_mu
rnn_sigma <- targets$rnn_sigma

sanity_check_obj <- function(gamma_vec) {
    gamma <- gamma_vec[1]
    
    # 1. Blended Logits
    blend_logits <- (1 - gamma) * calibrated_rnn_logits + gamma * bio_logits
    
    # 2. Precision-Weighted Kinematic Loss (Log-Normal KL Divergence style penalty)
    blend_mu <- (1 - gamma) * rnn_mu + gamma * bio_mu
    kinematic_loss <- mean( (rnn_mu - blend_mu)^2 / (2 * rnn_sigma^2) )
    
    # 3. Behavioral BCE against calibrated soft targets
    blend_probs <- plogis(blend_logits)
    # prevent strict 0/1 for log
    blend_probs <- pmax(pmin(blend_probs, 1 - 1e-7), 1e-7)
    bce <- -mean(p_val * log(blend_probs) + (1 - p_val) * log(1 - blend_probs))
    
    c(bce + kinematic_loss, 1.0 - gamma)
}

cat("Running NSGA-II Sanity Check with RNN Clone as Student...\n")
# Optimization parameter is just Gamma in [0, 1]
res <- nsga2(sanity_check_obj, idim = 1, odim = 2, 
             lower.bounds = c(0.0), upper.bounds = c(1.0), 
             popsize = 20, generations = 20)

best_idx <- which.min(res$value[,1])
cat("\n================ SANITY CHECK RESULTS ================\n")
cat(sprintf("Optimal Gamma chosen by NSGA-II: %.4f\n", res$par[best_idx, 1]))
cat(sprintf("Resulting BCE: %.4f\n", res$value[best_idx, 1]))
cat(sprintf("Resulting Teacher Penalty (1 - Gamma): %.4f\n", res$value[best_idx, 2]))
cat("========================================================\n")
