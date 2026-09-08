
library(dplyr)
df1 <- readr::read_csv("behavioral_compilate.csv", show_col_types = FALSE) %>%
	group_by(participant_id) %>% mutate(rt = (ttr - ttp) / 1000, iti = (ttp - lag(ttF)) / 1000) %>% ungroup()
sp1 <- unique(df1$participant_id)[1:100]
df_sub1 <- df1 %>% filter(participant_id %in% sp1) %>%
	mutate(iti = ifelse(is.na(iti) | iti < 0, median(iti, na.rm = TRUE), iti)) %>% filter(Resp %in% c(1, 2) & rt > 0.1)

df2 <- read.csv("behavioral_compilate.csv") %>%
	group_by(participant_id) %>% mutate(rt = (ttr - ttp) / 1000, iti = (ttp - lag(ttF)) / 1000) %>% ungroup()
sp2 <- unique(df2$participant_id)[1:100]
df_sub2 <- df2 %>% filter(participant_id %in% sp2) %>%
	mutate(iti = ifelse(is.na(iti) | iti < 0, median(iti, na.rm = TRUE), iti)) %>% filter(Resp %in% c(1, 2) & rt > 0.1)

cat("df_sub1 subj 6 trials:", sum(df_sub1$participant_id == levels(as.factor(df_sub1$participant_id))[6]), "\n")
cat("df_sub2 subj 6 trials:", sum(df_sub2$participant_id == levels(as.factor(df_sub2$participant_id))[6]), "\n")
cat("df_sub1 subj 52 trials:", sum(df_sub1$participant_id == levels(as.factor(df_sub1$participant_id))[52]), "\n")
cat("df_sub2 subj 52 trials:", sum(df_sub2$participant_id == levels(as.factor(df_sub2$participant_id))[52]), "\n")
