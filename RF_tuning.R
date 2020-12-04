rm(list = ls())
library(caret)
library(dplyr)
library(doParallel)
cl <- makePSOCKcluster(12)
registerDoParallel(cl)

source("Helper Functions.R")
df = read.csv("Original_data/P2Data2020.csv")
df = select(df,-X3,-X5,-X15)
df$Y = as.factor(df$Y)

custom =trainControl(method="repeatedcv",
                     number=10,
                     repeats=5,
                     verboseIter = T)
nodes = c(10,11,12,13,14,15,16)
all.MSEs = rep(NA, times = length(nodes))
all.RF = list(1:length(nodes)) 
tg <- data.frame(mtry = 3:12)
i = 1
for (this_node in nodes) {
  r.f <- train(Y~., df,
               method = "rf",
               tuneGrid = tg,
               nodesize= this_node,
               trControl=custom)
  all.MSEs[i] = max(r.f$results[,2])
  all.RF[[i]] = r.f
  i = i+1
  print(paste0(i," of ",length(nodes)))
}
all.RF[which.max(all.MSEs)]
r.f$results[,2]

# Best: node = 10,mtry = 3

stopCluster(cl)
