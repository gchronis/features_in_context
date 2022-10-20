library(tidyverse)

d = read_csv("swarm_predictions_tidy.csv")

d.s = group_by(d, preposition, verb, feauture) %>%
  summarise(m=mean(value)) %>%
  pivot_wider(id_cols=c(feauture, verb), names_from=preposition,
              values_from=m) %>%
  mutate(diff = `with` - `in`)

filter(d.s, verb == "creep") %>%
  arrange(diff) %>%
  filter(abs(diff) > .1)

filter(d.s, verb == "creep") %>%
  arrange(-diff) %>%
  filter(abs(diff) > .1)

filter(d.s, verb == "swim") %>%
  arrange(diff) %>%
  filter(abs(diff) > .1)

filter(d.s, verb == "swim") %>%
  arrange(-diff) %>%
  filter(abs(diff) > .1)


filter(d.s, verb == "swarm") %>%
  arrange(diff) 

filter(d.s, verb == "swarm") %>%
  arrange(-diff) 

filter(d.s, feauture == "is_round")

filter(d.s, feauture %in% c("has_patterns",
"is_large",
"a_container",
"used_for_holding_things",
"is_rough"))


#########
library(tidyverse)

d = read_csv("swarm_predictions_buchanan_minus_swim_tidy.csv")

d.s = group_by(d, preposition, verb, feauture) %>%
  summarise(m=mean(value)) %>%
  pivot_wider(id_cols=c(feauture, verb), names_from=preposition,
              values_from=m) %>%
  mutate(diff = `with` - `in`)

filter(d.s, verb == "creep") %>%
  arrange(diff) 
filter(d.s, verb == "creep") %>%
  arrange(-diff) 
filter(d.s, verb == "swarm") %>%
  arrange(diff)
filter(d.s, verb == "swarm") %>%
  arrange(-diff) 

filter(d.s, feauture %in% c("pattern",
                            "large",
                            "container",
                            "hold",
                            "rough"))

