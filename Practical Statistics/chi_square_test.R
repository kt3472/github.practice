library(ggplot2)
library(dplyr)
library(lmPerm)

click_rate <-  read.csv('./data/click_rates.csv')

## Chi square test

clicks <- matrix(click_rate$Rate, nrow=3, ncol=2, byrow=TRUE)

dimnames(clicks) <- list(unique(click_rate$Headline), unique(click_rate$Click))

chisq.test(clicks, simulate.p.value=TRUE)

chisq.test(clicks, simulate.p.value=FALSE)
