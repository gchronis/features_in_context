library(tidyverse)
library(ggrepel)
#d = read_csv("binder_feature_deltas_for_three_metaphors (1).csv")
d = read_csv("metaphor_predictions_tidy.csv")
cbbPalette <-
  c(
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7"
  )
rankedlevels = filter(d.s.sum, orig_subj == "nat. subj.")
  

d = mutate(d, Concrete = case_when(feature_category %in% c("Audition",
                                                           "Gustation",
                                                           "Motor",
                                                           "Olfaction",
                                                           "Somatic",
                                                           "Vision") ~ "Concrete",
                                   feature_category %in% c("Spatial", "Temporal") ~ 
                                     "SpatialTemporal",
                                   TRUE ~ "Abstract"),
           word = gsub("collapsed", "collapse", word))


d.sum = group_by(d, Concrete, feature_category, word) %>%
  summarise(mean.value = mean(value))
  
ggplot(d, aes(x=Concrete, y=value,
              fill=Concrete)) + 
geom_boxplot(alpha=.1) + 
    theme_classic(12) + 
  facet_grid(. ~ word) + 
  theme(legend.position = "none",
        axis.text.x = element_text(angle=90, vjust=.5, hjust=1)) + 
  scale_fill_manual(values=cbbPalette) +
  scale_colour_manual(values=cbbPalette) + 
  geom_text(size=3, 
            data=d.sum, aes(x=Concrete, y=mean.value,
                            label=feature_category)) +
  ylab("Literalness Minus Figurative Score") + 
  xlab("")
ggsave("literalness.png", width=5, height=6)

l = lmer(data=d, value ~ Concrete + (1|feauture) +
       (1 + Concrete|word) + 
       (1|feature_category))
l0 = lmer(data=d, value ~ 1 + (1|feauture) +
           (1 + Concrete|word) + 
           (1|feature_category))
summary(l)
anova(l, l0)


names(d) = c("Feature", "collapse", "cure", "flow")
d = gather(d, Word, value, -Feature)
d = left_join(d, bind)

ggplot(d, aes(x=Feature, y=value, group=Word, colour=Word)) +
  geom_point() + 
  theme_classic(12) + 
  theme(axis.text.x = element_text(angle=90, hjust=1, vjust=.5)) +
  ylab("Score")
ggsave("lineplot.png", width=6, height=5)

ggplot(d, aes(x=Word, y=value, label=Feature, colour=Concrete)) +
  geom_jitter(width=.05, alpha=.3) + 
  geom_text_repel(size=2) +
  theme_classic(12)
ggsave("textplot.png", width=7, height=5)
