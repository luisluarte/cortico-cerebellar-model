library(dplyr)
df <- read.csv('data/behavioral_compilate.csv')
p1 <- df[['participant_id']][1]
d <- df[df[['participant_id']] == p1, ]
d$diff_prob <- c(0, diff(d$prob))
rev_idx <- which(abs(d$diff_prob) > 0.4)
rev_window <- unique(unlist(lapply(rev_idx, function(x) seq(x, min(x+4, nrow(d))))))
stable_idx <- setdiff(1:nrow(d), rev_window)
cat('Total:', nrow(d), 'Rev:', length(rev_window), 'Stable:', length(stable_idx), '\n')
cat('Errors in Stable:', sum(d$F[stable_idx-1] == 0, na.rm=TRUE), '\n')
cat('Errors in Rev:', sum(d$F[rev_window-1] == 0, na.rm=TRUE), '\n')
