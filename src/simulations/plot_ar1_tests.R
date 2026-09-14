library(ggplot2)
library(dplyr)
library(patchwork)

long_df_all <- readRDS("results/long_df_all.rds")
safe_cor <- function(x) { 
  s <- sd(x, na.rm=TRUE)
  if(!is.na(s) && length(x) > 3 && s > 0) cor(x[-length(x)], x[-1], use="complete.obs") else NA_real_ 
}

# --- TEST 1 & 4: Continuous AR1 and Severe vs Healthy Cut ---
test14_df <- long_df_all %>%
  group_by(Subject, Condition) %>%
  summarize(
    Delta_Beta = first(Delta_Beta),
    AR1_Cont = safe_cor(P_Switch),
    .groups = "drop"
  ) %>% filter(!is.na(AR1_Cont))
  
# Based on the splines, < 0.35 is where the network collapses
test14_df$Severity <- ifelse(test14_df$Delta_Beta < 0.35, "Severe Damage (<0.35)", "Healthy / Mild (>0.35)")
test14_df$Severity <- factor(test14_df$Severity, levels=c("Healthy / Mild (>0.35)", "Severe Damage (<0.35)"))

p1 <- ggplot(test14_df, aes(x=Condition, y=AR1_Cont, fill=Condition)) +
  geom_boxplot(alpha=0.7, outlier.shape=NA, width=0.5) +
  geom_jitter(color="black", alpha=0.3, width=0.15) +
  facet_wrap(~Severity) +
  scale_fill_manual(values=c("Lesioned"="#d95f02", "Optimized"="#1b9e77")) +
  theme_bw(base_size=13) +
  labs(title="Test 1 & 4: Dilution & Continuous Data", 
       subtitle="Is the AR1 gap wider in severely damaged agents on continuous data?", 
       y="Continuous AR(1)", x="") +
  theme(legend.position="none", strip.text=element_text(face="bold"))

# --- TEST 3: Early vs Late Trials ---
test3_df <- long_df_all %>%
  mutate(Epoch = case_when(
    Trial <= 33 ~ "Early (1-33)",
    Trial > 67 ~ "Late (68-100)",
    TRUE ~ "Mid"
  )) %>%
  filter(Epoch != "Mid") %>%
  group_by(Subject, Condition, Epoch) %>%
  summarize(
    AR1_Cont = safe_cor(P_Switch),
    .groups = "drop"
  ) %>% filter(!is.na(AR1_Cont))
  
p3 <- ggplot(test3_df, aes(x=Condition, y=AR1_Cont, fill=Condition)) +
  geom_boxplot(alpha=0.7, outlier.shape=NA, width=0.5) +
  geom_jitter(color="black", alpha=0.3, width=0.15) +
  facet_wrap(~Epoch) +
  scale_fill_manual(values=c("Lesioned"="#d95f02", "Optimized"="#1b9e77")) +
  theme_bw(base_size=13) +
  labs(title="Test 3: Temporal Evolution (Cortex Catch-up)", 
       subtitle="Does the unoptimized cortex learn over time, masking the AR1 gap later?", 
       y="Continuous AR(1)", x="") +
  theme(legend.position="none", strip.text=element_text(face="bold"))
  
p_final <- p1 / p3 + plot_annotation(
  title="Testing Hypotheses for the AR1 Partition",
  theme=theme(plot.title=element_text(size=16, face="bold"))
)

ggsave("Fig47_AR1_Tests.png", p_final, width=8, height=10, dpi=300)
cat("Done.\n")
