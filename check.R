d <- readRDS('data/dataset_A.rds')
d_sub <- d[d$nt > 1, ]
cat(nrow(d_sub), "\n")
