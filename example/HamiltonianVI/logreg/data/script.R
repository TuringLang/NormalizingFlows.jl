library(readr)
library(matrixStats)

bank_full <- read_delim("bank-full.csv", 
                        delim = ";", escape_double = FALSE, 
                        trim_ws = TRUE)

bank_full$marital <- as.factor(bank_full$marital)
bank_full$marital <- as.numeric(bank_full$marital)
bank_full$housing <- as.factor(bank_full$housing)
bank_full$housing <- as.numeric(bank_full$housing)
bank_full$default <- as.factor(bank_full$default)
bank_full$default <- as.numeric(bank_full$default)

dat <- bank_full[,c(1,3,6,7,12,13,14,15,17)]
dat$y <- as.factor(dat$y)
dat$y <- as.numeric(dat$y)

dat <- dat[dat$marital < 3,]
dat <- dat[dat$housing < 3,]

dat$marital <- dat$marital - 1
dat$housing <- dat$housing - 1
dat$y <- dat$y - 1

inds <- 1:nrow(dat)
inds_1 <- inds[dat$y == 1]
inds_0 <- inds[dat$y == 0]

set.seed(519)

subsample_1 <- sample(1:length(inds_1), 200, replace=FALSE)
subsample_0 <- sample(1:length(inds_0), 200, replace=FALSE)

dat_1 <- dat[inds_1[subsample_1],]
dat_0 <- dat[inds_0[subsample_0],]

final_dat <- rbind(dat_1, dat_0)
final_dat_y <- final_dat$y

final_dat <- (final_dat - colMeans(final_dat)) / colSds(as.matrix(final_dat))
final_dat$y <- final_dat_y
final_dat$y <- as.factor(final_dat$y)

write.csv(final_dat, "final_dat.csv", row.names=TRUE)

model <- glm(data = final_dat, formula = y ~ ., family = binomial)
summary(model)

# > R.version
# _                           
# platform       x86_64-w64-mingw32          
# arch           x86_64                      
# os             mingw32                     
# system         x86_64, mingw32             
# status                                     
# major          4                           
# minor          1.2                         
# year           2021                        
# month          11                          
# day            01                          
# svn rev        81115                       
# language       R                           
# version.string R version 4.1.2 (2021-11-01)
# nickname       Bird Hippie