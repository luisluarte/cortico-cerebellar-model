
library(emoa)
set.seed(42)

gammas <- c(0.0, 0.25, 0.50, 0.75, 1.0)
models <- c("bio", "wsls", "qlearn")
B <- 1000

targets <- readRDS("src/r/distillation_targets_v2.rds")
N_subjs <- length(unique(targets$subjs))

union_front_pts <- c()

e_list <- list()
for(m in models) { e_list[[m]] <- list() }

# Load all data
for(g in gammas) {
    for(m in models) {
        if(m == "bio") {
            res <- readRDS(sprintf("results/res_bio_hypercost_gamma_%.2f.rds", g))
            t_bce <- res$value[, 1]
            t_kin <- res$value[, 2]
            e_val <- readRDS(sprintf("results/subj_errs_hypercost_gamma_%.2f.rds", g))
        } else if(m == "wsls") {
            res <- readRDS(sprintf("results/res_wsls_gamma_%.2f.rds", g))
            t_bce <- res$value[, 1]
            t_kin <- res$value[, 3]
            e_val <- readRDS(sprintf("results/subj_errs_wsls_gamma_%.2f.rds", g))
        } else {
            res <- readRDS(sprintf("results/res_qlearn_gamma_%.2f.rds", g))
            t_bce <- res$value[, 1]
            t_kin <- res$value[, 3]
            e_val <- readRDS(sprintf("results/subj_errs_qlearn_gamma_%.2f.rds", g))
        }
        
        v_bce <- apply(e_val[,,1, drop=FALSE], 1, mean)
        v_kin <- apply(e_val[,,2, drop=FALSE], 1, mean)
        
        local_pts <- rbind(t_bce, t_kin, v_bce, v_kin)
        local_nd <- nds_rank(local_pts) == 1
        union_front_pts <- cbind(union_front_pts, local_pts[, local_nd, drop=FALSE])
        
        e_list[[m]][[as.character(g)]] <- list(t_bce=t_bce, t_kin=t_kin, e_val=e_val)
    }
}

nadir <- apply(union_front_pts, 1, max)
nadir <- nadir * 1.05

# Bootstrapping Slopes
slopes_bio <- numeric(B)
slopes_wsls <- numeric(B)
slopes_qlearn <- numeric(B)
slopes_bio_log <- numeric(B)
slopes_wsls_log <- numeric(B)
slopes_qlearn_log <- numeric(B)

for(b in 1:B) {
    subj_idx <- sample(1:N_subjs, size=N_subjs, replace=TRUE)
    
    hv_vals <- list(bio=numeric(length(gammas)), wsls=numeric(length(gammas)), qlearn=numeric(length(gammas)))
    
    for(i in seq_along(gammas)) {
        g <- gammas[i]
        g_str <- as.character(g)
        
        for(m in models) {
            dat <- e_list[[m]][[g_str]]
            v_bce <- apply(dat$e_val[, subj_idx, 1, drop=FALSE], 1, mean)
            v_kin <- apply(dat$e_val[, subj_idx, 2, drop=FALSE], 1, mean)
            
            pts <- rbind(dat$t_bce, dat$t_kin, v_bce, v_kin)
            nd_idx <- nds_rank(pts) == 1
            front <- pts[, nd_idx, drop=FALSE]
            
            valid_front <- front[, front[1,] <= nadir[1] & front[2,] <= nadir[2] & front[3,] <= nadir[3] & front[4,] <= nadir[4], drop=FALSE]
            if(ncol(valid_front) > 0) {
                hv <- dominated_hypervolume(valid_front, nadir) / prod(nadir)
            } else {
                hv <- 0
            }
            hv_vals[[m]][i] <- hv
        }
    }
    
    fit_bio <- lm(hv_vals[["bio"]] ~ gammas)
    fit_wsls <- lm(hv_vals[["wsls"]] ~ gammas)
    fit_qlearn <- lm(hv_vals[["qlearn"]] ~ gammas)
    
    slopes_bio[b] <- coef(fit_bio)[2]
    slopes_wsls[b] <- coef(fit_wsls)[2]
    slopes_qlearn[b] <- coef(fit_qlearn)[2]

    # Log fits
    fit_bio_log <- lm(log(hv_vals[["bio"]] + 1e-4) ~ gammas)
    fit_wsls_log <- lm(log(hv_vals[["wsls"]] + 1e-4) ~ gammas)
    fit_qlearn_log <- lm(log(hv_vals[["qlearn"]] + 1e-4) ~ gammas)
    
    slopes_bio_log[b] <- coef(fit_bio_log)[2]
    slopes_wsls_log[b] <- coef(fit_wsls_log)[2]
    slopes_qlearn_log[b] <- coef(fit_qlearn_log)[2]

}

cat(sprintf("Bio Slope: %.4f [%.4f, %.4f]\n", mean(slopes_bio), quantile(slopes_bio, 0.025), quantile(slopes_bio, 0.975)))
cat(sprintf("WSLS Slope: %.4f [%.4f, %.4f]\n", mean(slopes_wsls), quantile(slopes_wsls, 0.025), quantile(slopes_wsls, 0.975)))
cat(sprintf("QLearn Slope: %.4f [%.4f, %.4f]\n", mean(slopes_qlearn), quantile(slopes_qlearn, 0.025), quantile(slopes_qlearn, 0.975)))

p_bio_vs_wsls <- mean(slopes_bio > slopes_wsls)
p_bio_vs_qlearn <- mean(slopes_bio > slopes_qlearn)

cat(sprintf("P(Bio Slope > WSLS Slope): %.4f\n", p_bio_vs_wsls))
cat(sprintf("P(Bio Slope > QLearn Slope): %.4f\n", p_bio_vs_qlearn))




cat("\n--- LOG SLOPES ---\n")
cat(sprintf("Bio Log-Slope: %.4f [%.4f, %.4f]\n", mean(slopes_bio_log), quantile(slopes_bio_log, 0.025), quantile(slopes_bio_log, 0.975)))
cat(sprintf("WSLS Log-Slope: %.4f [%.4f, %.4f]\n", mean(slopes_wsls_log), quantile(slopes_wsls_log, 0.025), quantile(slopes_wsls_log, 0.975)))
cat(sprintf("QLearn Log-Slope: %.4f [%.4f, %.4f]\n", mean(slopes_qlearn_log), quantile(slopes_qlearn_log, 0.025), quantile(slopes_qlearn_log, 0.975)))
cat(sprintf("P(Bio Log-Slope > WSLS Log-Slope): %.4f\n", mean(slopes_bio_log > slopes_wsls_log)))
cat(sprintf("P(Bio Log-Slope > QLearn Log-Slope): %.4f\n", mean(slopes_bio_log > slopes_qlearn_log)))

