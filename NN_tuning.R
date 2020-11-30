rm(list = ls())
library(nnet)
library(dplyr)


source("Helper Functions.R")
df = read.csv("Original_data/P2Data2020.csv")
df$Y = as.factor(df$Y)

X = df[,-1]
X = select(X,-X3,-X5,-X15)
Y = df[,1]
Y.num = class.ind(Y)

all.sizes = c(3,6,9,12,15,18)
all.decays = c(0, 0.001, 0.01, 0.1, 1)
all.pars = expand.grid(size = all.sizes, decay = all.decays)
n.pars = nrow(all.pars)
par.names = apply(all.pars, 1, paste, collapse = "-")

H = 3   # Number of times to repeat CV
K = 10   # Number of folds for CV
M = 20  # Number of times to re-run nnet() at each iteration

### Container for CV misclassification rates. Need to include room for
### H*K CV errors
CV.misclass = array(0, dim = c(H*K, n.pars))
colnames(CV.misclass) = par.names
start = Sys.time()
for(h in 1:H) {
  ### Get all CV folds
  folds = get.folds(nrow(X), K)
  
  for (i in 1:K) {
    print(paste0(h, "-", i, " of ", H, "-", K))
    
    
    ### Separate response from predictors
    Y.train.inner = Y[folds != i]
    X.train.inner.raw = X[folds != i, ]
    Y.valid.inner = Y[folds == i]
    X.valid.inner.raw = X[folds == i, ]
    
    ### Transform predictors and response for nnet()
    X.train.inner = rescale(X.train.inner.raw, X.train.inner.raw)
    X.valid.inner = rescale(X.valid.inner.raw, X.train.inner.raw)
    Y.train.inner.num = class.ind(factor(Y.train.inner))
    Y.valid.inner.num = class.ind(factor(Y.valid.inner))
    
    for (j in 1:n.pars) {
      ### Get parameter values
      this.size = all.pars[j, "size"]
      this.decay = all.pars[j, "decay"]
      
      ### Get ready to re-fit NNet with current parameter values
      MSE.best = Inf
      
      ### Re-run nnet() M times and keep the one with best sMSE
      for (l in 1:M) {
        this.nnet = nnet(
          X.train.inner,
          Y.train.inner.num,
          size = this.size,
          decay = this.decay,
          maxit = 2000,
          softmax = T,
          trace = F
        )
        this.MSE = this.nnet$value
        if (this.MSE < MSE.best) {
          MSE.best = this.MSE
          nnet.best = this.nnet
        }
      }
      
      ### Get CV misclassification rate for chosen nnet()
      pred.nnet.best = predict(nnet.best, X.valid.inner, type = "class")
      this.mis.CV = mean(Y.valid.inner != pred.nnet.best)
      
      ### Store this CV error. Be sure to put it in the correct row
      ind.row = (h - 1) * K + i
      CV.misclass[ind.row, j] = this.mis.CV
    }
  }
}
end = Sys.time()
### Make absolute and relative boxplots
boxplot(CV.misclass, las = 2)
rel.CV.misclass = apply(CV.misclass, 1, function(W) W/min(W))
boxplot(t(rel.CV.misclass), las=2,ylim = c(1,1.3))

