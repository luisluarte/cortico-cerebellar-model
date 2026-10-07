
library(emoa)
set.seed(42)

gammas <- c(0.0, 0.25, 0.50, 0.75, 1.0)
models <- c("bio", "wsls", "qlearn")
B <- 1000

targets <- readRDS("src/r/distillation_targets_v2.rds")
N_subjs <- length(unique(targets$subjs))

union_front_pts <- c()

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
    }
}

nadir <- apply(union_front_pts, 1, max)
nadir <- nadir * 1.05

cat(sprintf("Strict Global 4D Nadir (Union of Local Fronts): T_BCE=%.3f, T_Kin=%.3f, V_BCE=%.3f, V_Kin=%.3f\n", 
            nadir[1], nadir[2], nadir[3], nadir[4]))

results_df <- data.frame(Model=character(), Gamma=numeric(), HV_mean=numeric(), HV_lwr=numeric(), HV_upr=numeric())

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
        
        hv_dist <- numeric(B)
        for(b in 1:B) {
            subj_idx <- sample(1:N_subjs, size=N_subjs, replace=TRUE)
            v_bce <- apply(e_val[, subj_idx, 1, drop=FALSE], 1, mean)
            v_kin <- apply(e_val[, subj_idx, 2, drop=FALSE], 1, mean)
            
            pts <- rbind(t_bce, t_kin, v_bce, v_kin)
            nd_idx <- nds_rank(pts) == 1
            front <- pts[, nd_idx, drop=FALSE]
            
            valid_front <- front[, front[1,] <= nadir[1] & front[2,] <= nadir[2] & front[3,] <= nadir[3] & front[4,] <= nadir[4], drop=FALSE]
            if(ncol(valid_front) > 0) {
                hv_dist[b] <- dominated_hypervolume(valid_front, nadir)
            } else {
                hv_dist[b] <- 0
            }
        }
        
        norm_hv <- hv_dist / prod(nadir)
        results_df <- rbind(results_df, data.frame(
            Model=m, Gamma=g, 
            HV_mean=mean(norm_hv), HV_lwr=quantile(norm_hv, 0.025), HV_upr=quantile(norm_hv, 0.975)
        ))
    }
}
write.csv(results_df, "results/gamma_hv_bootstrap_4d_hypercost.csv", row.names=FALSE)
cat("Saved strict 4D CSVs.\n")


