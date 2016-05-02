# Load the libraries
library('vars')
library('urca')
library('pcalg')
library('dgof')
require(dgof)

# Read the input data
data.path <- "/home/zachncst/classes/csc591/projects/05.Topic-IV.Project-5.Manufacturer-Retailer-Price.CausalRelationDiscovery.distribution/Input Data/data.csv"
data.csv <- read.csv(data.path)
summary(data.csv)

# Build a VAR model 
var.model <- VAR(data.csv, p=1, type="const", ic="Schwartz")
summary(var.model)
var.model
plot(var.model)

# Extract the residuals from the VAR model 
var.residuals <- residuals(var.model)
var.residuals

# Check for stationarity using the Augmented Dickey-Fuller test 
move_dickey <- ur.df(var.residuals[,1])
rprice_dickey <- ur.df(var.residuals[,2])
mprice_dickey <- ur.df(var.residuals[,3])
summary(move_dickey)
summary(rprice_dickey)
summary(mprice_dickey)
#All values are less than the creitical value -1.95 at 5 percent (0.05)

# Check whether the variables follow a Gaussian distribution  
x <- rnorm(length(var.residuals))
move_test <- dgof::ks.test(var.residuals[,1], x, p.value=0.05)
rprice_kstest <- ks.test(var.residuals[,2], x, p.value=0.05)
mprice_kstest <- ks.test(var.residuals[,3], x, p.value=0.05)
move_test$p.value < 0.05
rprice_kstest$p.value < 0.05
mprice_kstest$p.value < 0.05

write.csv(var.residuals, file = "./residuals.csv", row.names = F)
# Write the residuals to a csv file to build causal graphs using Tetrad software

#Used the tetrad software but here is the code to print some of it

# PC algorithm
data <- var.residuals
suffStat=list(C=cor(data), n=1000)
pc_fit <- pc(suffStat, indepTest=gaussCItest, alpha=0.05, labels=colnames(data), skel.method="original")
pc_fit
plot(pc_fit, main="PC Output")

# LiNGAM algorithm
lingam_fit <- LINGAM(data)
show(lingam_fit)

