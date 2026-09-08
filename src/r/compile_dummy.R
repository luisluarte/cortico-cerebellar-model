
library(cmdstanr)
cat(Sys.getenv("CMDSTAN"))
mod <- cmdstan_model("dummy_bio.stan")
