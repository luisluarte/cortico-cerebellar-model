
library(loo)
res2 <- readRDS("results/exp2_B.rds")
res3 <- readRDS("results/exp3_B.rds")
res4 <- readRDS("results/exp4_B.rds")
baseline_fixed <- readRDS("results/q_w_fixed.rds")

models <- list(
    bio_unconst = res2$bio_unconst$loo,
    bio_anch = res4$bio_anch$loo,
    bio_locked = res3$bio_locked$loo,
    w_emp = baseline_fixed$w_emp$loo,
    q_emp = baseline_fixed$q_emp$loo
)

comp <- loo_compare(models)

cat("=================================================================\n")
cat("          FINAL FACTORIAL PSIS-LOO STATISTICAL REPORT            \n")
cat("=================================================================\n\n")

for(m_name in rownames(comp)) {
    m_loo <- models[[m_name]]
    elpd_loo <- m_loo$estimates["elpd_loo", "Estimate"]
    se_elpd <- m_loo$estimates["elpd_loo", "SE"]
    p_loo <- m_loo$estimates["p_loo", "Estimate"]
    k_warns <- sum(m_loo$diagnostics$pareto_k > 0.7)
    
    elpd_diff <- comp[m_name, "elpd_diff"]
    se_diff <- comp[m_name, "se_diff"]
    z_score <- ifelse(se_diff > 0, abs(elpd_diff / se_diff), 0)
    
    cat(sprintf("Model: %-13s\n", m_name))
    cat(sprintf("  Absolute ELPD : %8.1f (SE: %.1f)\n", elpd_loo, se_elpd))
    cat(sprintf("  ELPD Diff     : %8.1f (SE: %.1f) [Z = %.1f]\n", elpd_diff, se_diff, z_score))
    cat(sprintf("  Effective p_loo: %8.2f\n", p_loo))
    cat(sprintf("  Pareto k > 0.7 : %8d warnings\n", k_warns))
    cat("-----------------------------------------------------------------\n")
}

# Specific Hypothesis Tests (Z-scores of pairwise differences)
cat("\n--- Pairwise Hypothesis Tests (Z-scores) ---\n")
calc_z <- function(m1, m2, name1, name2) {
    comp_pair <- loo_compare(list(m1=m1, m2=m2))
    diff <- abs(comp_pair[2, "elpd_diff"])
    se <- comp_pair[2, "se_diff"]
    z <- diff / se
    cat(sprintf("%-12s vs %-12s: Delta ELPD = %5.1f (SE = %4.1f) | Z = %4.1f (p < 0.001: %s)\n", 
                name1, name2, diff, se, z, ifelse(z > 3.29, "YES", "NO")))
}

calc_z(models$bio_anch, models$w_emp, "bio_anch", "w_emp")
calc_z(models$bio_anch, models$q_emp, "bio_anch", "q_emp")
calc_z(models$bio_locked, models$w_emp, "bio_locked", "w_emp")
calc_z(models$bio_unconst, models$bio_anch, "bio_unconst", "bio_anch")


