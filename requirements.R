# requirements.R
# Consolidated installation script for Cortico-Cerebellar Model dependencies

cat("Installing required R packages for the Cortico-Cerebellar Model...\n")

# Setup CRAN mirror
local({r <- getOption("repos")
       r["CRAN"] <- "https://cloud.r-project.org"
       options(repos=r)
})

required_packages <- c(
  "Rcpp",
  "RcppArmadillo",
  "ggplot2",
  "dplyr",
  "tidyr",
  "zoo",
  "glmnet",
  "Matrix",
  "mgcv",
  "viridis",
  "ggpubr",
  "jsonlite",
  "pROC",
  "mco",
  "doParallel",
  "loo"
)

# Install CRAN packages if missing
for(pkg in required_packages) {
  if(!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
  }
}

# Install Torch for R
if(!require("torch", character.only = TRUE)) {
  install.packages("torch")
  library(torch)
  # Download LibTorch binaries
  install_torch()
}

cat("\nAll core dependencies successfully installed!\n")
