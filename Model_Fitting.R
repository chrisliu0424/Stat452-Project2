rm(list = ls())
library(FNN)
library(nnet)
library(glmnet)
library(mgcv)   # For GAM
library(klaR)   # For naive Bayes
library(randomForest)
library(dplyr)
library(e1071)

source("Helper Functions.R")
df = read.csv("Original_data/P2Data2020.csv")
df = dplyr::select(df,-X3,-X5,-X15)
df$Y = as.factor(df$Y)


X = df[,-1]
Y = df[,1]
Y.num = class.ind(Y)


data.matrix.raw = model.matrix(Y ~ ., data = df)
data.matrix = data.matrix.raw[,-1]

# set.seed(100)
V = 10    # 10-fold Cross-Validation
R = 5     # number of Replicates
n = nrow(X)
folds = get.folds(n,V) 
names = c("Knn-1","Knn-min","Knn-1se","Logistic","Logist-glmnet","Logist-LASSO-min","Logist-LASSO-1se","LDA","QDA",
          "GAM","NB:Normal","NB:Kernel","NB:Normal PC","NB:Kernel PC","RF","NN","SVM","Bagging")
misclass_df = matrix(NA,ncol = length(names), nrow = V*R)
colnames(misclass_df) <- names

current_row = 1
for (i in 1:R) {
  for(v in 1:V){
    print(paste0("r = ",i))
    print(paste0("v = ",v))
    train_df = df[folds!=v,]
    valid_df = df[folds==v,]
    train_matrix = model.matrix(Y ~ . ,train_df)
    valid_matrix = model.matrix(Y ~ . ,valid_df)
    
    train_df_scaled = cbind(train_df[,1],scale.1(train_df[,-1],train_df[,-1]))
    valid_df_scaled = cbind(valid_df[,1],scale.1(valid_df[,-1],train_df[,-1]))
    # Knn
    cv.knn = knn(train_df_scaled[,-1],valid_df_scaled[,-1],train_df_scaled[,1],k = 1)
    misclass_df[current_row,1] = mean(cv.knn != valid_df_scaled[,1])

    # Knn-min
    kmax <- 100
    k <- matrix(c(1:kmax), nrow=kmax)
    runknn <- function(x){
      knncv.fit <- knn.cv(train=train_df_scaled[,-1], cl=train_df_scaled[,1], k=x)
      # Fitted values are for deleted data from CV
      mean(ifelse(knncv.fit == train_df_scaled[,1], yes=0, no=1))
    }
    mis <- apply(X=k, MARGIN=1, FUN=runknn)
    mis.se <- sqrt(mis*(1-mis)/nrow(train_df_scaled)) #SE of misclass rates

    # Min rule
    mink = which.min(mis)
    knnfitmin.2 <- knn(train=train_df_scaled[,-1], test=valid_df_scaled[,-1], cl=train_df_scaled[,1], k=mink)
    misclass_df[current_row,2] = mean(knnfitmin.2!= valid_df_scaled[,1])

    # 1SE rule
    serule = max(which(mis<mis[mink]+mis.se[mink]))
    knnfitse.2 <- knn(train=train_df_scaled[,-1], test=valid_df_scaled[,-1], cl=train_df_scaled[,1], k=serule)
    misclass_df[current_row,3] = mean(knnfitse.2!= valid_df_scaled[,1])


    # Logistic
    train_df_scaled2 = data.frame(cbind(class = train_df[,1],rescale(train_df[,-1],train_df[,-1])))
    valid_df_scaled2 = data.frame(cbind(class = valid_df[,1],rescale(valid_df[,-1],train_df[,-1])))

    cv.logistic = nnet::multinom(class ~ ., data = train_df_scaled2,maxit = 2000)
    pred.logistic = predict(cv.logistic,newdata = valid_df_scaled2,type = "class")
    misclass_df[current_row,4] = mean(pred.logistic!= valid_df_scaled2$class)


    # Logist-LASSO(Logistic of Lasso version)
    logit.fit <- glmnet(x=as.matrix(train_df_scaled2[,-1]),
                        y=train_df_scaled2[,1], family="multinomial")
    logit.cv <- cv.glmnet(x=as.matrix(train_df_scaled2[,-1]),
                          y=train_df_scaled2[,1], family="multinomial")

    # glmnet
    pred.glmnet <- predict(logit.fit, s=0, type="class",
                              newx=as.matrix(valid_df_scaled2[,-1]))
    misclass_df[current_row,5] <- mean(pred.glmnet != valid_df_scaled2$class)

    # min
    pred.lasso.min <- predict(logit.fit, s=logit.cv$lambda.min, type="class",
                                newx=as.matrix(valid_df_scaled2[,-1]))
    misclass_df[current_row,6] <- mean(pred.lasso.min != valid_df_scaled2$class)

    # 1se
    pred.lasso.1se <- predict(logit.fit, s=logit.cv$lambda.1se, type="class",
                                newx=as.matrix(valid_df_scaled2[,-1]))
    misclass_df[current_row,7] <- mean(pred.lasso.1se != valid_df_scaled2$class)


    # LDA
    train_df_scaled3 <- data.frame(apply(train_df[,-1], 2, scale),class = train_df[,1])
    valid_df_scaled3 <- data.frame(apply(valid_df[,-1], 2, scale),class = valid_df[,1])
    cv.lda <- lda(data=train_df_scaled3, class~.)
    pred.lda <- predict(cv.lda, newdata=valid_df_scaled3)$class
    misclass_df[current_row,8] <- mean(pred.lda != valid_df_scaled3$class)


    # QDA
    cv.qda <- qda(data=train_df_scaled3, class~.)
    pred.qda <- predict(cv.qda, newdata=valid_df_scaled3)$class
    misclass_df[current_row,9] <- mean(pred.qda != valid_df_scaled3$class)
    
    
    # GAM
    train_df_num = train_df
    train_df_num$Y = as.numeric(as.factor(train_df_num$Y))-1
    ctrl <- list(epsilon = 1e-11,maxit = 1000)
    start = Sys.time()
    cv.gam1 <- gam(data=train_df_num, family = multinom(K=4),control=ctrl,
                   list(Y ~ +s(X6) +s(X9) +s(X14) ,
                        ~ s(X2) +s(X6) +s(X7) +s(X8) +s(X9) +s(X10),
                        ~ s(X6) +s(X7) +s(X8) +s(X9) +s(X10),
                        ~ s(X6) +s(X8) +s(X9) +s(X12))
                   )
    end1 = Sys.time()
    pred.gam.prob = predict(cv.gam1, valid_df, type = "response")
    end2 = Sys.time()
    pred.gam.1 = apply(pred.gam.prob, 1, which.max)
    # 
    misclass_df[current_row,10] <-  (mean(pred.gam.1!=as.numeric(as.factor(valid_df[,1]))))
    
    
    # NB
    # Normal, No PC
    cv.NB1 = NaiveBayes(train_df[,-1], train_df[,1], usekernel = F)
    pred.NB1 = predict(cv.NB1, valid_df)$class
    misclass_df[current_row,11] <- mean(pred.NB1 != valid_df[,1])

    # Kernel, No PC
    cv.NB2 = NaiveBayes(train_df[,-1], train_df[,1], usekernel = T)
    pred.NB2 = predict(cv.NB2, valid_df)$class
    misclass_df[current_row,12] <- mean(pred.NB2 != valid_df[,1])

    fit.PCA = prcomp(train_df[,-1], scale. = T)
    train_df.PC = fit.PCA$x  # Extract the PCs
    valid_df.PC = predict(fit.PCA, valid_df)

    # Normal, PC
    cv.NB3 = NaiveBayes(train_df.PC, train_df[,1], usekernel = F)
    pred.NB3 = predict(cv.NB3, valid_df.PC)$class
    misclass_df[current_row,13] <- mean(pred.NB3 != valid_df[,1])
    # Kernel, PC
    cv.NB4 = NaiveBayes(train_df.PC, train_df[,1], usekernel = T)
    pred.NB4 = predict(cv.NB4, valid_df.PC)$class
    misclass_df[current_row,14] <- mean(pred.NB4 != valid_df[,1])


    # Random Forest
    cv.rf <- randomForest(data=train_df, Y~.,nodesize = 10, mtry = 3)
    pred.rf = predict(cv.rf,valid_df)
    misclass_df[current_row,15] <-mean(pred.rf!= valid_df$Y)


    # NN
    train_df_scaled4 = data.frame(cbind(class = train_df[,1],rescale(train_df[,-1],train_df[,-1])))
    valid_df_scaled4 = data.frame(cbind(class = valid_df[,1],rescale(valid_df[,-1],train_df[,-1])))
    MSE.best = Inf    ### Initialize sMSE to largest possible value (infinity)
    M = 20            ### Number of times to refit.
    Y.train.num = class.ind(train_df_scaled4$class)
    for(k in 1:M){
      ### For convenience, we stop nnet() from printing information about
      ### the fitting process by setting trace = F.
      this.nnet = nnet(train_df_scaled4[,-1], Y.train.num, size = 8, decay = 0.01, maxit = 2000,
                       softmax = T, trace = F)
      this.MSE = this.nnet$value
      if(this.MSE < MSE.best){
        MSE.best = this.MSE
        cv.nn = this.nnet
      }
    }
    pred.nn = predict(cv.nn,newdata = valid_df_scaled4[,-1],type = "class")
    misclass_df[current_row,16] <-mean(pred.nn != valid_df$Y)


    # SVM
    cv.svm = svm(data = train_df, Y~., kernel = "radial", gamma = 10^(-3),cost = 10^(3))
    pred.svm = predict(cv.svm,newdata = valid_df)
    misclass_df[current_row,17] <-mean(pred.svm != valid_df$Y)

    temp = matrix(NA,ncol = 10,nrow = length(valid_df$Y))
    temp[,1] = knnfitmin.2
    temp[,2] = pred.logistic
    temp[,3] = as.numeric(as.factor(pred.lasso.min))
    temp[,4] = pred.lda
    temp[,5] = pred.qda
    temp[,6] = pred.NB2
    temp[,7] = pred.NB3
    temp[,8] = pred.rf
    temp[,9] = as.numeric(as.factor(pred.nn))
    temp[,10] = pred.svm

    pred.bag = c()
    for (ind in 1:nrow(temp)) {
      pred.bag[ind] = names(sort(summary(as.factor(temp[ind,])), decreasing=T)[1])
    }
    pred.bag.factor = as.factor(pred.bag)
    levels(pred.bag.factor) <- c("A","B","C","D","E")
    misclass_df[current_row,18] <- mean(pred.bag.factor != valid_df$Y)

    current_row = current_row + 1  
    }
}
# Relative Boxplot
# misclass_df = misclass_df[,-10]
low.s = apply(misclass_df, 1, min)
boxplot(misclass_df)
boxplot(misclass_df/low.s, las = 2, ylim = c(1,1.5),
        main=paste0("Plot for misclassification rate on ",V,"-folds validation"))


