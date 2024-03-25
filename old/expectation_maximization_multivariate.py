# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from tqdm import tqdm


def expectation_maximization(X, k, num_epochs=200, use_tqdm=True):
    n = X.shape[0]
    dim = X.shape[1]

    # shuffle the data
    X = X[np.random.permutation(n)]
    
    # Initialize parameters
    mu_hats = []
    sigma_hats = []
    pi_hats = []
    cov_of_X = np.cov(X.T) + 1
    for i in range(k):
        mu_hats.append(X[i])
        # sigma_hats.append(np.identity(dim))
        sigma_hats.append(cov_of_X)
        pi_hats.append(1 / k)

    # Perform EM algorithm for 20 epochs
    log_likelihoods = []
    if use_tqdm:
        tqdm_fn = tqdm
    else:
        tqdm_fn = lambda x: x

    for epoch in tqdm_fn(range(num_epochs)):
        # E-step: Compute responsibilities
        gammas = []
        total_gamma = np.zeros(n)
        for i in range(k):
            gamma = pi_hats[i] * multivariate_normal.pdf(X, mu_hats[i], sigma_hats[i])
            gammas.append(gamma)
            total_gamma += gamma

        for i in range(k):
            gammas[i] /= total_gamma
        
        # M-step: Update parameters
        for i in range(k):
            mu_hats[i] = np.linalg.multi_dot([gammas[i], X]) / gammas[i].sum()
            sigma_hats[i] = np.cov((X - mu_hats[i]).T, aweights=gammas[i])
            pi_hats[i] = np.mean(gammas[i])
        
        # Compute log-likelihood
        log_likelihood = 0
        for i in range(k):
            log_likelihood += pi_hats[i] * multivariate_normal.pdf(X, mu_hats[i], sigma_hats[i])
        log_likelihood = np.sum(np.log(log_likelihood))
        log_likelihoods.append(log_likelihood)
    return mu_hats, sigma_hats, pi_hats, log_likelihoods


if __name__ == '__main__':
    # Generate a dataset with two Gaussian components
    mu1, sigma1 = np.array([2, 1]), np.array([[0.5, 0.5], 
                                            [0.5, 2.]])
    mu2, sigma2 = np.array([1, 4]), np.array([[2.5, 1.5], 
                                            [1.5, 3.]])
    n = 1000
    X1 = np.random.multivariate_normal(mu1, sigma1, size=n, check_valid='warn', tol=1e-8)
    X2 = np.random.multivariate_normal(mu2, sigma2, size=n, check_valid='warn', tol=1e-8)
    X = np.concatenate([X1, X2])

    mu_hats, sigma_hats, pi_hats, log_likelihoods = expectation_maximization(X, 2, num_epochs=200, use_tqdm=True) 
    
    for i in range(2):
        p1 = multivariate_normal.pdf(X, mu_hats[i], sigma_hats[i])
        p2 = multivariate_normal.pdf(X1, mu_hats[i], sigma_hats[i])
        p3 = multivariate_normal.pdf(X2, mu_hats[i], sigma_hats[i])
        print(f"All: {round(p1.mean(), 4)}, 1: {round(p2.mean(), 4)}, 2: {round(p3.mean(), 4)}")

    plt.scatter(X1[:, 0], X1[:, 1], color='red')
    plt.scatter(X2[:, 0], X2[:, 1], color='green')
    plt.scatter(mu_hats[0][0], mu_hats[0][1], color='black', marker='x')
    plt.scatter(mu_hats[1][0], mu_hats[1][1], color='black', marker='x')

    plt.show()