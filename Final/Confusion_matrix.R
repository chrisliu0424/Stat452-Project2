rm(list = ls())
library(randomForest)
library(dplyr)

data = read.csv("Original_data/P2Data2020.csv")
data = dplyr::select(data,-X3,-X5,-X15)
data$Y = as.factor(data$Y)

p.train = 0.75
n = nrow(data)
n.train = floor(p.train*n)

ind.random = sample(1:n)
data.train = data[ind.random <= n.train,]
data.valid = data[ind.random > n.train,]
Y.valid = data.valid[,1]

rf.model <- randomForest(data=data.train, Y~.,nodesize = 10, mtry = 3)

rf.pred = predict(rf.model, data.valid)
table(Y.valid, rf.pred, dnn = c("Obs", "Pred"))
mean(Y.valid != rf.pred)
