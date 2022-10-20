library(tidyverse)
library(ggrepel)
d = read_csv("flow_collapse_and_cure_coarse_categories.csv")
names(d) = c("Feature", "collapse", "cure", "flow")
d = gather(d, Word, value, -Feature)

# lock in factor level order
d$Feature <- factor(d$Feature, levels = c('Vision', 'Somatic', 'Audition', 'Gustation', 'Olfaction',
                                          'Motor', 'Spatial', 'Temporal', 'Causal', 'Social',
                                          'Cognition', 'Emotion', 'Drive', 'Attention'))

ggplot(d, aes(x=Feature, y=value, group=Word, colour=Word)) +
  geom_point() + 
  geom_line() + 
  theme_classic(12) + 
  theme(axis.text.x = element_text(angle=90, hjust=1, vjust=.5)) +
  ylab("Score") +
  geom_hline(
    yintercept=0,
  )

ggplot(d, aes(x=Feature, y=value, group=Word, colour=Word)) +
  #geom_point() + 
  geom_col(position = "dodge") + 
  theme_classic(12) + 
  theme(axis.text.x = element_text(angle=90, hjust=1, vjust=.5)) +
  ylab("Score") +
  geom_hline(
    yintercept=0,
  )



ggplot(d, aes(x=Word, y=value, label=Feature)) +
  geom_point() + 
  geom_text_repel() +
  theme_classic(12)
