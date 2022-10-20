library(tidyverse)
library(ggrepel)
d = read.csv("~/Box Sync/src/features_in_context/r_visualizations/binder_type_level_map_rsquare.csv", header=TRUE)

ggplot(d, aes(x = Condition, y =rsquared, colour =Model, group = Model)) + stat_summary(fun = sum, geom = "line")
