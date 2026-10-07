
library(dplyr)
library(tidyr)

df <- readRDS("results/lesion_cortical_isolation.rds")

# Transform to wide format to easily pair by Subject
wide_df <- df %>%
    pivot_wider(names_from = Condition, values_from = Score) %>%
    rename(Intact = Intact, Lesioned = `Lesioned (No Thalamic Feedback)`) %>%
    mutate(Diff = Lesioned - Intact)

# Compute Paired Wilcoxon Signed-Rank Test and Medians
stats_df <- wide_df %>%
    group_by(Variable, Layer) %>%
    summarise(
        Median_Intact = median(Intact),
        Median_Lesioned = median(Lesioned),
        Median_Drop = median(Diff),
        P_Value = wilcox.test(Lesioned, Intact, paired=TRUE)$p.value,
        .groups = "drop"
    ) %>%
    mutate(FDR = p.adjust(P_Value, method="fdr")) %>%
    arrange(Variable, Layer)

cat("\n--- RAW STATISTICAL OUTPUT ---\n")
cat("| Variable | Layer | Intact (Median R2) | Lesioned (Median R2) | Drop (Delta) | P-value (FDR) |\n")
cat("|---|---|---|---|---|---|\n")
for (i in 1:nrow(stats_df)) {
    cat(sprintf("| %s | %s | %.3f | %.3f | **%.3f** | %.2e |\n",
        stats_df$Variable[i], stats_df$Layer[i],
        stats_df$Median_Intact[i], stats_df$Median_Lesioned[i],
        stats_df$Median_Drop[i], stats_df$FDR[i]))
}
cat("--- END STATISTICAL OUTPUT ---\n")

