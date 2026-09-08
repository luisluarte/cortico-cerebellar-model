
local_lib <- Sys.getenv("R_LIBS_USER")
.libPaths(c(local_lib, .libPaths()))
stan_data <- jsonlite::read_json("../../data/stan_data_N100_distilled.json", simplifyVector=TRUE)
print(summary(stan_data$RT))
