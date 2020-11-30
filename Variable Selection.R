library(glmnet)
library(randomForest)
rm(list = ls())
source("Helper Functions.R")
df = read.csv("Original_data/P2Data2020.csv")
df$Y = as.factor(df$Y)

X = df[,-1]
Y = df[,1]

data.matrix.raw = model.matrix(Y ~ ., data = df)
data.matrix = data.matrix.raw[,-1]

X_scaled = scale.1(X,X)

logit.fit <- glmnet(x=as.matrix(X), 
                    y=Y, family="multinomial")
logit.cv <- cv.glmnet(x=as.matrix(X), 
                      y=Y, family="multinomial")


# min
coef(logit.fit,s = logit.cv$lambda.min)

# 1se
coef(logit.fit,s = logit.cv$lambda.1se)



# RF

rf.model = randomForest(data = df, Y~.)
varImpPlot(rf.model)
rf.model$importance
