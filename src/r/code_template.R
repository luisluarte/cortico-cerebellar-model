# setup ----
# lib path
local_lib <- Sys.getenv("R_LIBS_USER")
if (!dir.exists(local_lib)) {
	dir.create(local_lib, recursive = TRUE)
}
.libPaths(c(local_lib, .libPaths()))

local({
	r <- getOption("repos")
	r["CRAN"] <- "https://cloud.r-project.org"
	options(repos = r)
})

# install libs
cat("##########################################################################")
cat("\ninstalling libs...\n")
cat("##########################################################################\n")
if (!require("pacman", character.only = TRUE)) {
	install.packages("pacman", lib = local_lib)
}

cat("##########################################################################")
cat("\nloading libs...\n")
cat("##########################################################################\n")
pacman::p_load(
	       torch,
	       jsonlite,
	       pROC
)

