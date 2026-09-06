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
	       tidyverse,
	       jsonlite,
	       this.path
)
setwd(here())

cat("##########################################################################")
cat("\nloading data...\n")
cat("##########################################################################\n")
df <- read_csv("../../data/behavioral_compilate.csv") %>%
	group_by(participant_id) %>%
	mutate(
	       rt = (ttr - ttp) / 1000,
	       iti = (ttp - lag(ttF)) / 1000
	) %>%
	ungroup()
selected_participants <- unique(df$participant_id)[1:100]
df_sub <- df %>%
	filter(participant_id %in% selected_participants) %>%
	mutate(
	       iti = ifelse(is.na(iti) | iti < 0, median(iti, na.rm = TRUE), iti)
	) %>%
	filter(Resp %in% c(1, 2) & rt > 0.1)
stan_data <- list(
		  N = nrow(df_sub),
		  N_subj = length(unique(df_sub$participant_id)),
		  subj =  as.numeric(as.factor(df_sub$participant_id)),
		  Bd1 = df_sub$Bd1,
		  Bd2 = df_sub$Bd2,
		  Resp = df_sub$Resp,
		  Reward = df_sub$F,
		  RT = df_sub$rt,
		  ITI = df_sub$iti
)
write_json(stan_data, "../../data/stan_data.json", auto_unbox = TRUE)
cat("Stan data ready\n")

