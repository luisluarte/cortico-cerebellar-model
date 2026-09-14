dA <- readRDS('data/dataset_A.rds')
t <- readRDS('src/r/distillation_targets_v2.rds')

p_names <- unique(dA$participant_id)
t_nums <- unique(t$subjs)

cat("Num names:", length(p_names), "\n")
cat("Num nums:", length(t_nums), "\n")

# Let's try to match the first name to ANY numeric id
found_any <- FALSE
for(i in 1:length(t_nums)) {
  s_num <- t_nums[i]
  idx <- which(t$subjs == s_num)
  tgt_labels <- t$labels[idx]
  
  for(j in 1:length(p_names)) {
    p_name <- p_names[j]
    d <- dA[dA$participant_id == p_name, ]
    resp <- d$Resp_mapped
    
    # Try offset
    for(offset in 0:10) {
      if(length(resp) >= length(idx) + offset) {
         if(all(tgt_labels == resp[(1+offset):(length(idx)+offset)])) {
            cat("Matched numeric", s_num, "to string", p_name, "with offset", offset, "\n")
            found_any <- TRUE
            break
         }
      }
    }
    if(found_any) break
  }
}
