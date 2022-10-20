library(tidyverse)
library(ggrepel)


d = read_csv("feature prediction results - token-level-eval-semcor-presentation-format.csv")

# lock in factor level order
d$Feature <- factor(d$ConcretenesRating, levels = c("0-1.5" ,  "1.5-2.5" ,"2.5-3.5", "3.5-4.5", "4.5-5" ))

# ggplot(d, aes(x="concreteness ratin", y="correlation", group=`model`, colour=`task`)) +
#   geom_point() + 
#   geom_line() + 
#   theme_classic(12) + 
#   theme(axis.text.x = element_text(angle=90, hjust=1, vjust=.5)) +
#   ylab("Correlation") +
#   geom_hline(
#     yintercept=0,
#   )

# Example Line plot with multiple groups
# ggplot(data=df2, aes(x=dose, y=len, group=supp)) +
#   geom_line()+
#   geom_point()

ggplot(d, aes(x=ConcretenesRating, y=Correlation, group=TaskModel, color=Task, linetype=Model, shape=Model)) +
  geom_point() +
  geom_line() + 
  xlab("Concreteness Bin") +
  ylab("Pearson Correlation") + 
  theme_classic(12) +
  theme(legend.position = c(0.87, 0.25),
        legend.background = element_rect(fill = "white", color = "black")) +
  #theme(aspect.ratio=11/16) + 
  theme(text = element_text(size = 20))  
  #theme(axis.text.x = element_text(angle=90, hjust=1, vjust=.5)) +
  #ylab("Correlation") +
  #geom_hline(
  #  yintercept=0,
  # )
