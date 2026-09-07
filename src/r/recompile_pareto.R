
library(tibble)
res_wsls <- readRDS("fast_res_wsls.rds")
res_qlearn <- readRDS("fast_res_qlearn.rds")
res_bio <- readRDS("fast_res_bio.rds")

tib <- tibble(
  Model = c(rep("WSLS", nrow(res_wsls$value)), rep("QLearn", nrow(res_qlearn$value)), rep("Bio", nrow(res_bio$value))),
  NLL = c(res_wsls$value[,1], res_qlearn$value[,1], res_bio$value[,1]),
  Teacher_Penalty = c(res_wsls$value[,2], res_qlearn$value[,2], res_bio$value[,2]),
  Gamma = 1 - c(res_wsls$value[,2], res_qlearn$value[,2], res_bio$value[,2]),
  Pareto_Optimal = c(res_wsls$pareto.optimal, res_qlearn$pareto.optimal, res_bio$pareto.optimal)
)

saveRDS(tib, "distillation_2_fast_pareto.rds")
cat("Successfully added Pareto_Optimal column!\n")
