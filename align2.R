df <- read.csv('data/behavioral_compilate.csv')
t <- readRDS('src/r/distillation_targets_v2.rds')

s1 <- unique(t$subjs)[1]
idx <- which(t$subjs == s1)

cat("Target labels for s1:\n")
print(head(t$labels[idx], 20))

found <- FALSE
for(p in unique(df$participant_id)) {
  d <- df[df$participant_id == p, ]
  # In 01_preprocess, labels are Resp_mapped (Resp - 1)
  resp <- d$Resp - 1
  
  # Try to find the substring of labels in resp
  # Since trials might be truncated at the beginning
  for(offset in 0:5) {
    if(length(resp) >= length(idx) + offset) {
      if(all(t$labels[idx] == resp[(1+offset):(length(idx)+offset)])) {
         cat("Matched numeric subj", s1, "to string", p, "with offset", offset, "\n")
         found <- TRUE
         break
      }
    }
  }
  if(found) break
}
if(!found) cat("Could not match labels!\n")
