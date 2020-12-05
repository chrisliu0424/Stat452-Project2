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

trcon = trainControl(method="repeatedcv", number=10, repeats=2,
                     returnResamp="all")
parmgrid = expand.grid(C=10^c(0:5), sigma=10^(-c(5:0)))

tuned.nnet <- train(x=df[,-1], y=df$Y, method="svmRadial", 
                    preProcess=c("center","scale"), trace=FALSE, 
                    tuneGrid=parmgrid, trControl = trcon)


resamples = reshape(data=tuned.nnet$resample[,-2], idvar=c("C", "sigma"), 
                    timevar="Resample", direction="wide")

C.sigma <- paste(log10(resamples[,1]),"-",log10(resamples[,2]))
best = apply(X=resamples[,-c(1,2)], MARGIN=2, FUN=max)

boxplot.matrix(x=t(t(1-resamples[,-c(1:2)])), use.cols=FALSE, names=C.sigma,
               main="Misclassification rates for different Cost-Gamma", las=2)

boxplot.matrix(x=t(t(1-resamples[,-c(1:2)])/(1-best)), use.cols=FALSE, names=C.sigma,
               main="Relative Misclass rates for different Cost-Gamma", las=2)

par(mfrow=c(1,2))
boxplot(t(t(1-resamples[,-c(1:2)])/(1-best)) ~ resamples[,1], xlab="C", ylab="Relative Error")
boxplot(t(t(1-resamples[,-c(1:2)])/(1-best)) ~ resamples[,2], xlab="Sigma", ylab="Relative Error")


stopCluster(cl)

# Best C = 10^3, sigma = 10^-3