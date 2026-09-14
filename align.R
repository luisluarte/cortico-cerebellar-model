df <- read.csv('data/behavioral_compilate.csv')
t <- readRDS('src/r/distillation_targets_v2.rds')

# For the first numeric subj in targets
s1 <- unique(t$subjs)[1]
idx <- which(t$subjs == s1)

# Sequence of Bd1 and Bd2 (from X)
# In X, Bd1 is at index 1, Bd2 is at index 2 (scaled). Wait, in targets$X, what are the columns?
# Let's print the first 5 rows of X for s1
cat("Target X for s1:\n")
print(head(t$X[idx, 1:4]))

# Find a participant in df that matches this sequence
found <- FALSE
for(p in unique(df$participant_id)) {
  d <- df[df$participant_id == p, ]
  # In 01_preprocess, Bd1_scaled = (Bd1 - min) / (max - min)
  # But we also have lag_reward in X[ ,3] maybe?
  # Let's just compare lag_reward.
  if(nrow(d) >= length(idx)) {
      # lag_reward in targets is t$lag_reward.
      if(all(t$lag_reward[idx] == d$F[1:length(idx)]) || 
         all(t$lag_reward[idx] == d$F[2:(length(idx)+1)])) {
         cat("Matched numeric subj", s1, "to string", p, "\n")
         found <- TRUE
         break
      }
  }
}
if(!found) cat("Could not match!\n")
