library(ggplot2)
library(dplyr)
library(lmerTest)

df <- readRDS("results/perseveration_df.rds")
# Note: df has Trial, Subject, Condition, Prev_Choice, Prev_Feedback, State.
# But we need the actual Choice and Feedback at trial t to compute the Bayesian sequence!
# Fortunately, we can re-extract it from the raw empirical data.
df_emp <- read.csv("data/behavioral_compilate.csv", stringsAsFactors = FALSE)

# Bayesian Ideal Observer function
run_ideal_observer <- function(choices, rewards, p_reward=0.8, hazard=0.05) {
  # choices: 1 or 2
  # rewards: 1 (win) or 0 (loss)
  n <- length(choices)
  p_s1 <- numeric(n)
  
  # Initialize at 0.5
  current_p1 <- 0.5
  
  for(t in 1:n) {
    # 1. Prediction (incorporate hazard rate)
    pred_p1 <- current_p1 * (1 - hazard) + (1 - current_p1) * hazard
    
    # 2. Update (incorporate likelihood of current trial's outcome)
    c <- choices[t]
    r <- rewards[t]
    
    if(!is.na(c) && !is.na(r) && c %in% c(1,2)) {
      if(c == 1) {
        lik1 <- ifelse(r == 1, p_reward, 1 - p_reward)
        lik2 <- ifelse(r == 1, 1 - p_reward, p_reward)
      } else {
        lik1 <- ifelse(r == 1, 1 - p_reward, p_reward)
        lik2 <- ifelse(r == 1, p_reward, 1 - p_reward)
      }
      
      unnorm1 <- pred_p1 * lik1
      unnorm2 <- (1 - pred_p1) * lik2
      current_p1 <- unnorm1 / (unnorm1 + unnorm2)
    } else {
      current_p1 <- pred_p1
    }
    
    p_s1[t] <- current_p1
  }
  return(p_s1)
}

cat("1. Running Bayesian Ideal Observer on all empirical sequences...\n")
df_opt_list <- list()
subjects <- unique(df$Subject)
for(subj in subjects) {
  d_s <- df_emp %>% filter(participant_id == subj, Resp %in% c(1,2), ((ttr - ttp) / 1000) > 0.1)
  p_ideal <- run_ideal_observer(d_s$Resp, d_s$F)
  # Map back to trials 2:100
  trial_nums <- 2:nrow(d_s)
  
  # For the model's choices, P_Switch is relative to Choice t-1.
  # So P(Choice t = 1) = if Choice_t-1 == 1 { 1 - P_Switch } else { P_Switch }
  # Let's rebuild P(Choice t = 1) for the models:
  d_mod <- df %>% filter(Subject == subj)
  # Merge the ideal observer
  # The ideal observer outputs P(S=1).
  
  ideal_df <- data.frame(Subject=subj, Trial=trial_nums, P_Ideal_1 = p_ideal[trial_nums], Choice_T = d_s$Resp[trial_nums])
  df_opt_list[[subj]] <- ideal_df
}
ideal_all <- bind_rows(df_opt_list)

# Join with model df
df <- df %>% left_join(ideal_all, by=c("Subject", "Trial"))

# Reconstruct Model P(Choice = 1)
# P_Switch is probability of switching FROM Prev_Choice
df$P_Mod_1 <- ifelse(df$Prev_Choice == 1, 1 - df$P_Switch, df$P_Switch)

# Calculate Deviation from Optimal
df$Dev_Opt <- abs(df$P_Mod_1 - df$P_Ideal_1)

# Now evaluate Deviation from Optimal as a function of Loss_Streak!
df_loss <- df %>% filter(Loss_Streak > 0)
df_loss$Loss_Streak_Cap <- ifelse(df_loss$Loss_Streak > 4, "5+", as.character(df_loss$Loss_Streak))
df_loss$Severity <- ifelse(df_loss$Delta_Beta < 0.35, "Severe Damage (< 0.35)", "Healthy / Mild (> 0.35)")

agg_dev <- df_loss %>%
  group_by(Condition, Loss_Streak_Cap, Severity) %>%
  summarize(
    Mean_Dev = mean(Dev_Opt, na.rm=TRUE),
    SE = sd(Dev_Opt, na.rm=TRUE) / sqrt(n()),
    .groups = "drop"
  )

agg_dev$Loss_Streak_Cap <- factor(agg_dev$Loss_Streak_Cap, levels = c("1", "2", "3", "4", "5+"))

p <- ggplot(agg_dev, aes(x = Loss_Streak_Cap, y = Mean_Dev, color = Condition, group = Condition)) +
  geom_line(linewidth = 1.5) +
  geom_point(size = 4) +
  geom_errorbar(aes(ymin = Mean_Dev - SE, ymax = Mean_Dev + SE), width = 0.2, linewidth = 1) +
  facet_wrap(~ Severity) +
  scale_color_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  theme_bw(base_size = 15) +
  labs(
    title = "Deviation from Bayes-Optimal Inference",
    subtitle = "Absolute error between the neural network and a theoretical Ideal Observer",
    x = "Consecutive Losses on the Same Choice (Loss Streak)",
    y = "Deviation from Optimal Probability"
  ) +
  theme(legend.position = "bottom", strip.text=element_text(face="bold"))

ggsave("Fig53_Deviation_Ideal_Observer.png", p, width = 10, height = 6, dpi = 300)

cat("\n--- Frequentist Test: Deviation from Optimal (Severe Group) ---\n")
m_dev <- lmer(Dev_Opt ~ Condition * Loss_Streak + (1|Subject), data=df_loss %>% filter(Delta_Beta < 0.35))
print(anova(m_dev))
cat("\nDone.\n")
