
local_lib <- Sys.getenv("R_LIBS_USER")
.libPaths(c(local_lib, .libPaths()))
library(torch)
state <- torch_load("../../results/frozen_rnn_baseline.pt", device = "cpu")
print(names(state))
