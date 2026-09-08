
local_lib <- Sys.getenv("R_LIBS_USER")
.libPaths(c(local_lib, .libPaths()))
ps <- readRDS("parameter_stability.rds")
if(is.data.frame(ps)) {
    mu_alpha_pc <- ps$alpha_pc[3]
} else {
    mu_alpha_pc <- ps$Bio$alpha_pc[3]
}
print(mu_alpha_pc)
print(ps$Bio)
