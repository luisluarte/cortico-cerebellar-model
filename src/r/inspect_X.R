
targets <- readRDS("distillation_targets_v2.rds")
X <- targets$X
cat("Dimensions of X:", dim(X), "\n")
if (!is.null(colnames(X))) {
    cat("Column names of X:\n")
    print(colnames(X))
} else {
    cat("X does not have column names. Looking at a single row to infer:\n")
    print(X[1, ])
}

# If it doesn't have col names, I might need to look at how it was generated, or just inspect the values
