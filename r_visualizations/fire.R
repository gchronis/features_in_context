library(tidyverse)

d = read_csv("../data/processed/prototype_predictions_buchanan_fire_tidy.csv")
cl = read_csv("../data/processed/bnc_clusters_for_fire.csv")

dnn = read_csv("../data/processed/prototype_predictions_buchanan_fire_ffnn_tidy.csv")



# filter(d.s, word == "fire") %>%
#   arrange(diff) %>%
#   filter(abs(diff) > .1)
# 
# filter(d.s, verb == "creep") %>%
#   arrange(-diff) %>%
#   filter(abs(diff) > .1)
# 
# filter(d.s, verb == "swim") %>%
#   arrange(diff) %>%
#   filter(abs(diff) > .1)
# 
# filter(d.s, verb == "swim") %>%
#   arrange(-diff) %>%
#   filter(abs(diff) > .1)




d.s = group_by(d, prototype_id, word, feauture) %>%
  summarise(m=mean(value)) %>%
  pivot_wider(id_cols=c(feauture, word), names_from=prototype_id,
              values_from=m) %>%
  mutate(diff = `1` - `3`)


filter(d.s) %>%
  arrange(diff) 

filter(d.s) %>%
  arrange(-diff) 

file = '../data/processed/top_buchanan_features_fire.csv'
# look at top features for cluster 0
filter(d.s) %>%
  subset(select = -c(`1`, `2`, `3`, `4`))  %>%
  arrange(-`0`) 
  #write.table(x, file = “”, append = FALSE, quote = TRUE, sep = ” “, row.names = TRUE, col.names = TRUE)

# look at top features for cluster 1
filter(d.s) %>%
  subset(select = -c(`0`, `2`, `3`, `4`))  %>%
  arrange(-`1`) 

filter(d.s) %>%
  subset(select = -c(`0`,`1`, `3`, `4`))  %>%
  arrange(-`2`) 

# cluster 3
filter(d.s) %>%
  subset(select = -c(`0`,`1`, `2`, `4`))  %>%
  arrange(-`3`) 

filter(d.s) %>%
  subset(select = -c(`0`,`1`, `2`, `3`))  %>%
  arrange(-`4`) 

# look at top and bottom differences between cluster 1 dif cluster 4
filter(d.s) %>%
  subset(select = -c(`0`, `2`, `4`) )  %>%
  arrange(diff)
  

filter(d.s) %>%
  subset(select = -c(`0`, `2`, `4`) )  %>%
  arrange(-diff)  

# try to 2 way anova by cluster ID and category
# summary(d)
# one.way <- aov(value ~ feauture, data = d)
# summary(one.way)

# to characterize how each one is distinct from the others, average the others and then compare the diff
filter(d.s) %>%
  # 34567
  # 01234
  # you want to compare 0 to the rest
  mutate(other = rowMeans(d.s[ , c(4,5,6,7)], na.rm=TRUE)) %>%
  mutate(diff = `0` - other) %>%
  arrange(-diff) %>%
  write.table(file = "transformative_fire.csv", sep = ",", quote = FALSE, row.names = F)

# to characterize how each one is distinct from the others, average the others and then compare the diff
filter(d.s) %>%
  # 34567
  # 01234
  # you want to compare 1 to the rest
  mutate(other = rowMeans(d.s[ , c(3,5,6,7)], na.rm=TRUE)) %>%
  mutate(diff = `1` - other) %>%
  arrange(-diff) %>%
  write.table(file = "destructive_fire.csv", sep = ",", quote = FALSE, row.names = F)

# to characterize how each one is distinct from the others, average the others and then compare the diff
filter(d.s) %>%
  # 34567
  # 01234
  # you want to compare 2 to the rest
  mutate(other = rowMeans(d.s[ , c(3,4,6,7)], na.rm=TRUE)) %>%
  mutate(diff = `2` - other) %>%
  arrange(-diff) %>%
  write.table(file = "gun_fire.csv", sep = ",", quote = FALSE, row.names = F)

# to characterize how each one is distinct from the others, average the others and then compare the diff
filter(d.s) %>%
  # 34567
  # 01234
  # you want to compare 2 to the rest
  mutate(other = rowMeans(d.s[ , c(3,4,5,7)], na.rm=TRUE)) %>%
  mutate(diff = `3` - other) %>%
  arrange(-diff) %>%
  write.table(file = "cooking_fire.csv", sep = ",", quote = FALSE, row.names = F)

# to characterize how each one is distinct from the others, average the others and then compare the diff
filter(d.s) %>%
  # 34567
  # 01234
  # you want to compare 4 to the rest
  mutate(other = rowMeans(d.s[ , c(3,4,5,6)], na.rm=TRUE)) %>%
  mutate(diff = `4` - other) %>%
  arrange(-diff) %>%
  write.table(file = "controlled_fire.csv", sep = ",", quote = FALSE, row.names = F)


filter(d.s, feauture %in% c("destroy"TRUEfilter(d.s, feauture %in% c("destroy",
"bad",
"hurt",
"destroy",
"burn",
"cook",
"danger",
"kill",
"weapon",
"metal",
"person",
"light",
"red",
"heat",
"control",
"law",
"city",
"gun",
"energy",
"eat",
"weapon",
"work",
"police",
"crime",
"law",
"area",
"group",
"force"
))


##########
# run for ffnnn


dnn.s = group_by(dnn, prototype_id, word, feauture) %>%
  summarise(m=mean(value)) %>%
  pivot_wider(id_cols=c(feauture, word), names_from=prototype_id,
              values_from=m) %>%
  mutate(diff = `1` - `3`)

filter(dnn.s) %>%
  arrange(diff) 

filter(dnn.s) %>%
  arrange(-diff) 

file = '../data/processed/top_buchanan_features_fire.csv'
# look at top features for cluster 0
filter(dnn.s) %>%
  subset(select = -c(`1`, `2`, `3`, `4`))  %>%
  arrange(-`0`) 
#write.table(x, file = “”, append = FALSE, quote = TRUE, sep = ” “, row.names = TRUE, col.names = TRUE)

# look at top features for cluster 1
filter(dnn.s) %>%
  subset(select = -c(`0`, `2`, `3`, `4`))  %>%
  arrange(-`1`) 

filter(dnn.s) %>%
  subset(select = -c(`0`,`1`, `3`, `4`))  %>%
  arrange(-`2`) 

# cluster 3
filter(dnn.s) %>%
  subset(select = -c(`0`,`1`, `2`, `4`))  %>%
  arrange(-`3`) 

filter(dnn.s) %>%
  subset(select = -c(`0`,`1`, `2`, `3`))  %>%
  arrange(-`4`) 

# look at top and bottom differences between cluster 1 dif cluster 4
filter(dnn.s) %>%
  subset(select = -c(`0`, `2`, `4`) )  %>%
  arrange(diff)


filter(dnn.s) %>%
  subset(select = -c(`0`, `2`, `4`) )  %>%
  arrange(-diff)  


filter(dnn.s, feauture %in% c("destroy",
                            "bad",
                            "hurt",
                            "destroy",
                            "burn",
                            "cook",
                            "danger",
                            "kill",
                            "weapon",
                            "metal",
                            "person",
                            "light",
                            "red",
                            "heat",
                            "control",
                            "law",
                            "city",
                            "gun",
                            "energy",
                            "eat",
                            "weapon",
                            "work",
                            "police",
                            "crime",
                            "law",
                            "area",
                            "group",
                            "force"
))

#########
library(tidyverse)

dmc = read_csv("../data/processed/prototype_predictions_mcrae_fire_tidy.csv")

dmc.s = group_by(dmc, prototype_id, word, feauture) %>%
  summarise(m=mean(value)) %>%
  pivot_wider(id_cols=c(feauture, word), names_from=prototype_id,
              values_from=m) %>%
  mutate(diff = `1` - `4`)


# biggest diffs between destructive fire and hearth fire
filter(dmc.s) %>%
  arrange(diff) 
filter(dmc.s) %>%
  arrange(-diff) 

# look at top and bottom features for cluster 1 (destructive)
filter(dmc.s) %>%
  subset(select = c(feauture, word, `1`))  %>%
  arrange(-`1`)
# filter(dmc.s) %>%
#   subset(select = c(feauture, word, `1`))  %>%
#   arrange(`1`)


# look at top and bottom features for cluster 3 (cooking)
filter(dmc.s) %>%
  subset(select = c(feauture, word, `3`))  %>%
  arrange(-`3`)

# look at top and bottom features for cluster 4 (control)
filter(dmc.s) %>%
  subset(select = c(feauture, word, `4`))  %>%
  arrange(-`4`)


filter(d.s, feauture %in% c(
  "used_for_cooking",
  "found_in_kitchens",
  "an_appliance",
  inbeh_-_produces_heat
))
                            
