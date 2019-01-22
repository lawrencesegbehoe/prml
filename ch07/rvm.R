## Final Project 

################################################
#######                                  #######
#######         RVM classification       #######
#######                                  #######
################################################

# Two groups CLASSIFICATION problem

# A matrix function
A.fun   <- function(a){diag(c(a))}

# gradient function
g.fun   <- function(PHI, tN, yN, A, w){t(PHI)%*%(tN - yN) - A%*%w}

# Hessian function
H.fun   <- function(PHI,B,A){-(t(PHI)%*%PHI + A)}
# H.fun   <- function(PHI,B,A){-(t(PHI)%*%B%*%PHI + A)}

# gamma function
gam.fun <- function(a,H){1 - a*diag(-solve(H))}

# sigmoid function
sig.fun <- function(x){1/(1+exp(-x))}

# B matrix function
B.fun   <- function(yn){diag(c(yn*(1 - yn)))}

# function for updating W
Newton_Raphson.fun = function(a, w, PHI, tN, w_delta){
  a_old         <- a    # initial alpha vector
  w_old         <- w    # initial weight vector
  w_Thresh      <- 5e-4 # Threshold below which the loop breaks for 
                        # distance between one w and the next w
  
  ## Fix a_old and update w_old 
  while (w_delta > w_Thresh) {
    
    yn <- sig.fun(PHI%*%w_old)     # yn
    A  <- A.fun(a_old)             # A matrix  
    g  <- g.fun(PHI,tN,yn,A,w_old) # gradient vector
    B  <- B.fun(yn)                # B matrix
    H  <- H.fun(PHI,B,A)           # Hessian matrix
    
    # Newton Raphson IRLS
    w_new <- w_old - solve(H)%*%g
    
    ## Criterion for stopping the IRLS
    w_delta <- norm(w_new-w_old,"F")
    
    ## update w_old
    w_old <- w_new
  }
  
  return(list(w_new = w_new,# weight vector
              H     = H,    # Hessian vector
              yn    = yn
              )
         )
  
}

## Main function - RMV Classifier

RVM.classifier.fun <- function(a, w, PHI, tN, w_delta, a_delta){
  
  a_old         <- a    # initial alpha vector
  w_old         <- w    # initial weight vector
  delta_Thresh  <- 5e-4 # Threshold below which the loop breaks for 
                        # distance between one alpha and the next alpha
  
  a_Inf_Thresh  <- 1e9  # Inf threshold
  
  while(a_delta > delta_Thresh){
    
    ## Fix a_old and update w_old 
    result.NR  <- Newton_Raphson.fun(a_old, w_old, PHI, tN, w_delta)
    
    w_new <- result.NR$w_new # get weight vector
    H     <- result.NR$H     # get Hessian matrix
    yn    <- result.NR$yn
    
    ## Fix w_new and update a_old 
    
    gamma_i <- gam.fun(a_old, H) # get gamma
    a_new   <- gamma_i/(w_new^2) # get new alpha vector
    
    ## convert alpha beyond the Inf thresh hold
    a_new[a_new > a_Inf_Thresh] <- a_Inf_Thresh
    
    ## convergence criterion
    a_delta <- norm(a_new-a_old,"F")
    
    ## Just a check of the distance between
    ## one alpha and the next alpha    
    print(a_delta)
    
    a_old  <- a_new # update alpha
    w_old  <- w_new # update weight
  }
  return(list(a_new = a_new,
              w_new = w_new,
              yn    = yn))
}

########################################################

# get the given dataset
load("ink.training.rdata")
# check dimension
dim(ink.training.dat)

# 2D sketchy visualization
plot(ink.training.dat[,1, 2,1], type = "n")
legend("topleft",fill = c(2,4), #cex = 0.8,
       legend = c("Class 1", "Class 2"), border = NA)
for(i in 1:22){
lines(ink.training.dat[,1, i,1], col =2)
lines(ink.training.dat[,1, i,2], col =4)
}
## convert given dataset into a single matrix of 6200 by 44

## get container for that matrix
mat1 <- matrix(NA, ncol = 22, nrow = 200*31)  # for class 1
mat2 <- matrix(NA, ncol = 22, nrow = 200*31)  # for class 2

for (i in 1:22) {mat1[,i] <- ink.training.dat[ , , i, 1]} # class 1
for (i in 1:22) {mat2[,i] <- ink.training.dat[ , , i, 2]} # class 2

ink_dat <- cbind(mat1,mat2) # combined matrix of 6200 by 44

saveRDS(ink_dat,file = "inkdat.Rda") # save it for convenience

inkdat <- readRDS("inkdat.Rda") # load it back into the workspace
# dim(inkdat)  #[1] 6200   44
# class(inkdat)#[1] "matrix"
# str(inkdat)  #num [1:6200, 1:44] 81.1 91 75.8 17.4 17.3 ...

#######################################################


## Correlation Kernel is used

## get container for the Gram matrix 
cor.mat = matrix(NA, ncol = dim(inkdat)[2], nrow = dim(inkdat)[2])

## do a loop for the Gram matrix
for (i in 1:dim(inkdat)[2]) {
  for(j in 1:dim(inkdat)[2])
    cor.mat[i,j] <- cor(inkdat[,i], inkdat[, j])
}



PHI <- cor.mat # put Gram matrix into PHI 

N <- dim(inkdat)[2] # dimension of 44 objects

## initial values
w   <- rep(0.3, dim(PHI)[2])
a   <- rep(0.3, dim(PHI)[2])
tN  <- rep(0:1, each = 22)

# run the RVM classifier 
RVM.classifier.result = RVM.classifier.fun(a, w, PHI, tN, w_delta = 2, a_delta = 3)

# get the weight and alpha
w_new <- RVM.classifier.result$w_new
a_new <- RVM.classifier.result$a_new
yn    <- RVM.classifier.result$yn

## get relevant vectors
ix.rvm = which(a_new < 1e9)
w_rvm <- w_new[ix.rvm]

## Make predicitons
## Using the training data set as test data
Gram_pred = matrix(NA, nrow = dim(inkdat)[2], ncol = length(ix.rvm))
inkdat.rv = inkdat[, ix.rvm]

for (i in 1:dim(inkdat)[2]){
  for(j in 1:length(ix.rvm)){
    Gram_pred[i,j] <- cor(inkdat[,i], inkdat.rv[, j])
  }
}

RVM.prob = sig.fun(Gram_pred%*%w_rvm)
RVM.pred = ifelse(RVM.prob < 0.5,0,1)
table(RVM.pred, True.classes = tN)


###### Done ########




