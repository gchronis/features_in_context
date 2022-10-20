library(tidyverse)
library(lme4)
library(lmvar)

d = read_csv("swapped_ffnn.csv") %>%
  filter(
    id != "email-enronsent36_02-0075\n",
    id != "weblog-blogspot.com_healingiraq_20050121235804_ENG_20050121_235804-0014\n"
  )
d$subjobj = ifelse(
  d$subject > d$object,
  paste0(d$subject, "_", d$object),
  paste0(d$object, "_", d$subject)
)
d$id = paste0(d$id, "_", d$subjobj)


bind = read_csv("binder_feature_names.csv") %>%
  rename(feature = feature_name) %>%
  select(feature_category, feature)

d = left_join(d, bind)

d = rename(d, swap = `swapped?`)


d.sum = group_by(d, feature, condition, swap) %>%
  summarize(m = mean(predicted_value)) %>%
  pivot_wider(
    id_cols = c(feature),
    names_from = c(condition, swap),
    values_from = c(m)
  ) %>%
  rename(
    nat.obj = object_FALSE,
    subj.forced.to.obj = object_TRUE,
    nat.subj = subject_FALSE,
    obj.forced.to.subj = subject_TRUE
  )

d.s =  pivot_wider(
  d,
  id_cols = c(id, feature,
              feature_category),
  names_from = c(condition, swap),
  values_from = c(predicted_value)
) %>%
  rename(
    nat.obj = object_FALSE,
    subj.forced.to.obj = object_TRUE,
    nat.subj = subject_FALSE,
    obj.forced.to.subj = subject_TRUE
  )

d.s$nat.subjectness = d.s$nat.subj - d.s$nat.obj
group_by(d.s, feature) %>%
  summarise(nat.subjectness.mean = mean(nat.subjectness)) %>%
  arrange(-nat.subjectness.mean)

group_by(d.s, feature) %>%
  summarise(nat.subjectness.mean = mean(nat.subjectness)) %>%
  arrange(nat.subjectness.mean)

#
d.s$forced.subj = d.s$obj.forced.to.subj - d.s$nat.obj
d.s$forced.obj = d.s$subj.forced.to.obj - d.s$nat.subj


group_by(d.s, feature) %>%
  summarise(m = mean(forced.subj)) %>%
  arrange(-m)

group_by(d.s, feature) %>%
  summarise(m = mean(forced.subj)) %>%
  arrange(m)


group_by(d.s, feature) %>%
  summarise(m = mean(forced.obj)) %>%
  arrange(-m)

group_by(d.s, feature) %>%
  summarise(m = mean(forced.obj)) %>%
  arrange(m)

ggplot(d.s, aes(x = nat.obj, y = obj.forced.to.subj)) + geom_point() +
  geom_smooth() + facet_wrap( ~ feature_category, ncol = 2)


ggplot(d.s, aes(x = nat.subj, y = subj.forced.to.obj)) + geom_point() +
  geom_smooth() + facet_wrap( ~ feature_category, ncol = 2)



ggplot(d.s, aes(x = nat.subj, y = subj.forced.to.obj)) + geom_point() +
  geom_smooth() + facet_wrap( ~ feature_category, ncol = 2)

cor(d.s$nat.subj, d.s$subj.forced.to.obj)
cor(d.s$nat.subj, d.s$obj.forced.to.subj)
cor(d.s$nat.obj, d.s$subj.forced.to.obj)
cor(d.s$nat.obj, d.s$obj.forced.to.subj)


# when we push an object to be a subject, what increases and by how much?
obj.to.subj = mutate(d.s, diff = obj.forced.to.subj - nat.obj) %>%
  group_by(feature_category, feature) %>%
  summarise(mean.diff = mean(diff),
            pct.inc = mean(diff > 0))
arrange(obj.to.subj,-mean.diff)
arrange(obj.to.subj, mean.diff)

ids.that.move = filter(d.s, feature == "Human", obj.forced.to.subj - nat.obj > 2) %>%
  select(id)
filter(d, id %in% ids.that.move$id, swap == F) %>%
  select(subject, object) %>%
  unique()


ids.that.move = filter(d.s, feature == "Number", obj.forced.to.subj - nat.obj < -1) %>%
  select(id)
filter(d, id %in% ids.that.move$id, swap == F) %>%
  select(subject, object) %>%
  unique()



# when we push a subject to be an object, what increases and by how much?
subj.to.obj = mutate(d.s, diff = subj.forced.to.obj - nat.subj) %>%
  group_by(feature_category, feature) %>%
  summarise(mean.diff = mean(diff),
            pct.inc = mean(diff > 0))
arrange(subj.to.obj,-mean.diff)
arrange(subj.to.obj, mean.diff)


# regressions
summary(lm(data = d, predicted_value ~ condition * feature))
summary(lm(data = filter(d, swap == T), predicted_value ~ condition * feature))


# this is for no swaps, compare if subjs and objs differ
# bonferroni says divide by m
m = length(unique(d$feature))
m
print(.05/m )
summary(l <- lm(data = filter(d, swap == F), predicted_value ~ condition * feature))

# in paper, for context
summary(lint <- lm(data = d, predicted_value ~ condition * feature * swap))
cc = summary(lint)$coefficients
tibble(coef=rownames(cc), p=cc[, 4], val=cc[, 1] > 0) %>%
  filter(p < .05/m,
         grepl("swapTRUE", coef),
         grepl("conditionsubject", coef)) %>%
  select(coef) %>%
  data.frame()

##########################
summary(lm(data = d, predicted_value ~ condition * feature_category))
summary(lm(
  data = filter(d, swap == T),
  predicted_value ~ condition * feature_category
))
summary(lm(
  data = filter(d, swap == F),
  predicted_value ~ condition * feature_category
))


############
d.s.sum = group_by(d.s, feature_category) %>%
  mutate(tot.mean = (
    sum(nat.subj) + sum(nat.obj) + sum(obj.forced.to.subj) +
      sum(subj.forced.to.obj)
  ) / (4 * n())) %>%
  summarize(
    nat.subj.mean = mean(nat.subj - tot.mean),
    nat.obj.mean = mean(nat.obj - tot.mean),
    obj.forced.to.subj = mean(obj.forced.to.subj - tot.mean),
    subj.forced.to.obj = mean(subj.forced.to.obj - tot.mean)
  ) %>%
  gather(variable, value,-feature_category) %>%
  mutate(
    orig_subj = ifelse(
      variable == "nat.subj.mean" | variable == "subj.forced.to.obj",
      "nat. subj.",
      "nat. obj."
    ),
    variable = ifelse(grepl("forced", variable), "forced", "natural")
  ) %>%
  spread(variable, value) #, -feature_category, -orig_subj)

cbbPalette <-
  c(
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7"
  )
rankedlevels = filter(d.s.sum, orig_subj == "nat. subj.") %>%
  arrange(natural)
d.s.sum$feature_category = factor(d.s.sum$feature_category,
                                  levels = rankedlevels$feature_category)
ggplot(
  d.s.sum,
  aes(
    x = feature_category,
    xend = feature_category,
    colour = orig_subj,
    group = orig_subj,
    y = natural,
    yend = forced
  )
)  +
  geom_segment(arrow = arrow(length = unit(0.3, "cm")),
               size = 1,
               alpha = .9) +
  theme_classic(12)  +
  theme(legend.title = element_blank()) +
  coord_flip() +
  scale_colour_manual(values = cbbPalette) +
  xlab("Feature Category") +
  ylab("(centered) predicted value") +
  geom_point()

####################3
d.s.sum = group_by(d.s, feature) %>%
  mutate(tot.mean = (
    sum(nat.subj) + sum(nat.obj) + sum(obj.forced.to.subj) +
      sum(subj.forced.to.obj)
  ) / (4 * n())) %>%
  summarize(
    nat.subj.mean = mean(nat.subj - tot.mean),
    nat.obj.mean = mean(nat.obj - tot.mean),
    obj.forced.to.subj = mean(obj.forced.to.subj - tot.mean),
    subj.forced.to.obj = mean(subj.forced.to.obj - tot.mean)
  ) %>%
  gather(variable, value,-feature) %>%
  mutate(
    orig_subj = ifelse(
      variable == "nat.subj.mean" | variable == "subj.forced.to.obj",
      "nat. subj.",
      "nat. obj."
    ),
    variable = ifelse(grepl("forced", variable), "forced", "natural")
  ) %>%
  spread(variable, value) #, -feature, -orig_subj)

cbbPalette <-
  c(
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7"
  )
rankedlevels = filter(d.s.sum, orig_subj == "nat. subj.") %>%
  arrange(natural)
d.s.sum$feature = factor(d.s.sum$feature,
                         levels = rankedlevels$feature)
ggplot(
  d.s.sum,
  aes(
    x = feature,
    xend = feature,
    colour = orig_subj,
    group = orig_subj,
    y = natural,
    yend = forced
  )
)  +
  geom_segment(arrow = arrow(length = unit(0.3, "cm")),
               size = .9,
               alpha = .9) +
  theme_classic(12)  +
  theme(legend.title = element_blank(),
        legend.position = c(.8, .1)) +
  coord_flip() +
  scale_colour_manual(values = cbbPalette) +
  xlab("Feature Category") +
  ylab("(centered) predicted value") +
  geom_point()
ggsave("subj_obj_features.png", width=4, height=8)

###### get the most

filter(d, feature == "Biomotion", swap == F) %>%
  arrange(-predicted_value) %>%
  select(condition, predicted_value, subject, object)

filter(d, feature == "Biomotion", swap == F) %>%
  arrange(predicted_value)





########
trainnum = 300
set.seed(42)
train_ids = sample(unique(l1.data$id))[1:trainnum]
set.seed(42)
test_ids = sample(unique(l1.data$id))[(trainnum + 1):length(unique(l1.data$id))]

############
d$Answer = d$condition == "subject"
l1.data = filter(d, swap == F)

# this function does backoff logistic regression on a trainin gset
# then cross-val on a test set
do.crossval <- function(l1.data) {
  l1.data.spread = select(l1.data, id, feature, predicted_value, Answer) %>%
    spread(feature, predicted_value)
  
  l1.data.train  = filter(l1.data.spread, id %in% train_ids) %>%
    select(-id)
  l1.data.test = filter(l1.data.spread, id %in% test_ids) %>%
    select(-id)
  
  l1 = glm(family = "binomial",
           data = l1.data.train,
           Answer ~  .,
           x = T)
  l0 = glm(family = "binomial",
           data = l1.data.train,
           Answer ~  1,
           x = T)
  
  l1step = step(l0, scope = list(lower = l0, upper = l1), direction="forward")
  test.lm = glm(family = "binomial",
                data = l1.data.test,
                l1step$formula,
                x = T)
  acc <- function(r, pi)
    mean(abs(r - pi) <  0.5)
  a = cv.glm(l1.data.test, test.lm, cost = acc, K = 5)
  print(a$delta)
}

d$Answer = d$condition == "subject"
l1.data = filter(d, swap == F)
do.crossval(l1.data)

d$Answer = d$condition == "subject"
l2.data = filter(d, swap == T)
do.crossval(l2.data)

l3.data = filter(d, condition == "subject")
l3.data$Answer = l3.data$swap
do.crossval(l3.data)

l4.data = filter(d, condition == "object")
l4.data$Answer = l4.data$swap
do.crossval(l4.data)

