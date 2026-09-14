library(brms)
library(ggplot2)
library(dplyr)
library(patchwork)

# Load the Spline Model
fit_spline <- readRDS("results/brms_switch_model_calibrated_simplified.rds")

# We want to extract the conditional effects specifically for the Reversal State!
# The simplified model formula was: P_Calibrated ~ Condition + State + s(Delta_Beta, by = Condition)
# This means the slope of Delta_Beta is identical in Stable and Reversal, but the intercept shifts!
# Wait, if the slope is identical, we aren't estimating the Reversal-specific slope.
