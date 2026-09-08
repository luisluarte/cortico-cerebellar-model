
local_lib <- Sys.getenv("R_LIBS_USER")
.libPaths(c(local_lib, .libPaths()))
ps <- readRDS("parameter_stability.rds")
if(is.data.frame(ps)) {
    q3_bio <- subset(ps, Model == "Bio" & Quartile == "Q3")
} else {
    q3_bio <- subset(ps$Bio, Quartile == "Q3")
}
print(q3_bio)
