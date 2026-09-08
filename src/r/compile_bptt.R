
local_lib <- Sys.getenv("R_LIBS_USER")
.libPaths(c(local_lib, .libPaths()))

suppressPackageStartupMessages({
  library(cmdstanr)
})

cat("Compiling Full Hierarchical Bio Stan Model with BPTT (allow-undefined)...\n")
mod_full <- tryCatch({
  cmdstan_model("full_bio_bptt.stan", 
                stanc_options = list("allow-undefined"=TRUE),
                cpp_options = list(USER_HEADER = file.path(getwd(), "bio_bptt.hpp")))
}, error = function(e) {
  cat("Compilation failed!\n")
  cat(e$message, "\n")
})
