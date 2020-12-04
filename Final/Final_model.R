rm(list = ls())
library(randomForest)
library(dplyr)

train_df = read.csv("Original_data/P2Data2020.csv")
test_df = read.csv("Original_data/P2Data2020testX.csv")
train_df = dplyr::select(train_df,-X3,-X5,-X15)
test_df = dplyr::select(test_df,-X3,-X5,-X15)
train_df$Y = as.factor(train_df$Y)

rf.model <- randomForest(data=train_df, Y~.,nodesize = 10, mtry = 3)
prediction = predict(rf.model,newdata = test_df)
write.table(prediction, "Final/prediction.csv", sep = ",", row.names = F, col.names =F)
