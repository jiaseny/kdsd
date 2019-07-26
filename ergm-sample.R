# Draw samples from an undirected ERGM distribution and save to text file
# Requires the statnet R package suite:
# https://cran.r-project.org/web/packages/statnet/index.html)

# Parse command line args
args = commandArgs(trailingOnly = TRUE)
n = as.integer(args[1])  # Number of network samples
d = as.integer(args[2])  # Number of nodes in each network
rho = as.numeric(args[3])  # Edge parameter
theta = as.numeric(args[4])  # Two-star parameter
tau = as.numeric(args[5])  # Triangle parameter
seed = as.integer(args[6])  # Random seed
res_dir = as.character(args[7])  # Directory for storing results

print(sprintf("n = %d, d = %d, rho = %.3f, theta = %.3f, tau = %.3f, seed = %d, res_dir = %s", n, d, rho, theta, tau, seed, res_dir))


suppressMessages(library(statnet))
set.seed(seed)

p = d*(d-1)/2  # Number of possible edges
coef <- c(rho, theta, tau)

sample_ergm <- function(coef) {
  g <- simulate(network(d, directed=FALSE) ~ edges + kstar(2) + triangle,
                nsim=1, coef=coef)
  A <- as.matrix(g)
  x <- A[upper.tri(A, diag = FALSE)]

  return(x)
}

# Actually draws 2*n samples since MMD requires n samples each from p and q
#   while KDSD only uses n samples
X = matrix(nrow=2*n, ncol=p)  # Store results
for (i in 1:(2*n)) {
  if (i %% 100 == 1) {
    print(sprintf("Sampled %d graphs ...", i))
  }
  X[i,] = sample_ergm(coef)
}

# Save samples to text file
fname = sprintf("ergm-samples-n%d-d%d-rho%.3f-theta%.3f-tau%.3f-seed%d.txt", n, d, rho, theta, tau, seed)
write.table(X, paste(res_dir, fname, sep=""),
            row.names = FALSE, col.names = FALSE)
