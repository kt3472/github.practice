library(ggplot2)
library(dplyr)
library(lmPerm)

click_rate <-  read.csv('./data/click_rates.csv')
imanishi <-  read.csv('./data/imanishi_data.csv')

## Chi square test

clicks <- matrix(click_rate$Rate, nrow=3, ncol=2, byrow=TRUE)

dimnames(clicks) <- list(unique(click_rate$Headline), unique(click_rate$Click))

chisq.test(clicks, simulate.p.value=TRUE)

chisq.test(clicks, simulate.p.value=FALSE)

x <- seq(1, 30, length=100)
chi <- data.frame(df = factor(rep(c(1, 2, 5, 10), rep(100, 4))),
                  x = rep(x, 4),
                  p = c(dchisq(x, 1), dchisq(x, 2), dchisq(x, 5), dchisq(x, 20)))

ggplot(chi, aes(x=x, y=p)) +
  geom_line(aes(linetype=df)) +
  theme_bw() +
  labs(x='', y='')

dev.off()

## Fishers exact test
fisher.test(clicks)

imanishi$Digit <- factor(imanishi$Digit)
ggplot(imanishi, aes(x=Digit, y=Frequency)) +
  geom_bar(stat="identity") +
  theme_bw()
  
dev.off()



