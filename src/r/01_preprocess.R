
cat("Starting 01_preprocess.R...\n")

data_raw <- read.csv("../../../data/behavioral_compilate.csv")

# Keep only valid responses
d <- data_raw[data_raw$Resp %in% c(1, 2), ]

# Calculate RT in seconds
d$RT_sec <- (d$ttr - d$ttp) / 1000
d <- d[d$RT_sec > 0.05 & d$RT_sec < 30, ] # Filter crazy RTs

# Map RT to 0-1
d$RT_scaled <- (d$RT_sec - min(d$RT_sec)) / (max(d$RT_sec) - min(d$RT_sec))

# Map Bd1 and Bd2 to 0-1
min_bd <- min(c(d$Bd1, d$Bd2))
max_bd <- max(c(d$Bd1, d$Bd2))
d$Bd1_scaled <- (d$Bd1 - min_bd) / (max_bd - min_bd)
d$Bd2_scaled <- (d$Bd2 - min_bd) / (max_bd - min_bd)

# Map Resp to 0-1
d$Resp_mapped <- d$Resp - 1

# Feedback
d$F_mapped <- d$F

# Calculate ITI
d$ITI_sec <- 0
for(p in unique(d$participant_id)) {
  idx <- which(d$participant_id == p)
  if(length(idx) > 1) {
    itis <- (d$ttp[idx[-1]] - d$ttF[idx[-length(idx)]]) / 1000
    # Clean ITIs
    itis <- pmax(pmin(itis, 20), 0) 
    d$ITI_sec[idx[-1]] <- itis
    d$ITI_sec[idx[1]] <- mean(itis, na.rm=TRUE)
  }
}
d$ITI_scaled <- (d$ITI_sec - min(d$ITI_sec)) / (max(d$ITI_sec) - min(d$ITI_sec))

# Create lag features
d$lag_Reward <- 0
d$lag_Ch <- 0
d$lag_Resp <- 0
for(p in unique(d$participant_id)) {
  idx <- which(d$participant_id == p)
  if(length(idx) > 1) {
    d$lag_Reward[idx[-1]] <- d$F_mapped[idx[-length(idx)]]
    d$lag_Resp[idx[-1]] <- d$Resp_mapped[idx[-length(idx)]]
    # lag_Ch maps 1 to 0 and 2 to 1 based on BD chosen. 
    # Let's say Choice is left or right. We can just use Resp.
    d$lag_Ch[idx[-1]] <- d$Resp_mapped[idx[-length(idx)]]
  }
}

# Select top 100 participants
counts <- sort(table(d$participant_id), decreasing=TRUE)
top_100 <- names(counts)[1:100]
d_top100 <- d[d$participant_id %in% top_100, ]

set.seed(42)
subj_A <- sample(top_100, 50)
subj_B <- setdiff(top_100, subj_A)

data_A <- d_top100[d_top100$participant_id %in% subj_A, ]
data_B <- d_top100[d_top100$participant_id %in% subj_B, ]

saveRDS(d_top100, "../../data/dataset_D.rds")
saveRDS(data_A, "../../data/dataset_A.rds")
saveRDS(data_B, "../../data/dataset_B.rds")
saveRDS(subj_A, "../../data/subj_A.rds")
saveRDS(subj_B, "../../data/subj_B.rds")

cat("Preprocessing complete!\n")
