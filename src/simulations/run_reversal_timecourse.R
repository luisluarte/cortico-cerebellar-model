library(dplyr)
library(mgcv)
library(ggplot2)

df <- readRDS("results/reversal_df.rds")
emp <- read.csv("data/behavioral_compilate.csv")

# Filter down to the vulnerable half (where lesions actually hurt) to see the policy failure clearly
median_db <- median(unique(df$Delta_Beta))
df <- df %>% filter(Delta_Beta <= median_db)

cat("Aligning trials to Reversals...\n")
aligned_list <- list()
for(s in unique(df$Subject)) {
  s_emp <- emp[emp$participant_id == s, ]
  s_df <- df[df$Subject == s, ]
  
  if(nrow(s_emp) > 0) {
    # Find reversal trials (where probability flips)
    s_emp$prob_diff <- c(0, diff(s_emp$prob))
    rev_trials <- s_emp$nt[abs(s_emp$prob_diff) > 0.4] # 'nt' is trial number in the block? No, wait.
    # What is the trial column? Let's use 1:nrow(s_emp) since we just saw 'nt' in head()
    rev_trials <- which(abs(s_emp$prob_diff) > 0.4)
    
    s_df$Time_Since_Rev <- NA
    for(i in 1:nrow(s_df)) {
      t <- s_df$Trial[i]
      past_revs <- rev_trials[rev_trials <= t]
      if(length(past_revs) > 0) {
        s_df$Time_Since_Rev[i] <- t - max(past_revs)
      }
    }
    aligned_list[[length(aligned_list) + 1]] <- s_df
  }
}

df_aligned <- bind_rows(aligned_list)
# Filter to 0-20 trials post-reversal
df_window <- df_aligned %>% filter(Time_Since_Rev >= 0 & Time_Since_Rev <= 20)

cat(sprintf("N window trials: %d\n", nrow(df_window)))

df_window$Condition <- as.factor(df_window$Condition)

cat("Fitting GAM...\n")
m_gam <- gam(P_Switch_Sq ~ Condition + s(Time_Since_Rev, by = Condition, k = 7), 
             data = df_window, family = betar(link="logit"))

nd <- expand.grid(
  Time_Since_Rev = seq(0, 20, by = 0.5),
  Condition = factor(c("Optimized", "Lesioned"))
)
preds <- predict(m_gam, newdata = nd, type = "response", se.fit = TRUE)

nd$fit <- preds$fit
nd$lower <- preds$fit - 1.96 * preds$se.fit
nd$upper <- preds$fit + 1.96 * preds$se.fit

p <- ggplot(nd, aes(x = Time_Since_Rev, y = fit, color = Condition, fill = Condition)) +
  geom_line(linewidth = 1.5) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, color = NA) +
  scale_color_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  scale_fill_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  theme_bw(base_size = 15) +
  labs(
    title = "Policy Maintenance After Environmental Reversal",
    subtitle = "GAM of P(Switch) [Vulnerable Half of Population]",
    x = "Trials Since Environmental Reversal",
    y = "P(Switch | Loss)"
  ) +
  theme(legend.position = "bottom")

ggsave("Fig66_Reversal_Timecourse_GAM.png", p, width = 8, height = 6, dpi = 300)
cat("Done.\n")
