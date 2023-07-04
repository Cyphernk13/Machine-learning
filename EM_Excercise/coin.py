import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

# Function to perform the expectation-maximization algorithm for estimating coin biases
def coin_em(rolls, theta_A=None, theta_B=None, maxiter=10):
    # Initializing theta_A and theta_B with random values
    theta_A = theta_A or random.random()
    theta_B = theta_B or random.random()
    thetas = [(theta_A, theta_B)]  # List to store the values of theta_A and theta_B at each iteration

    # Perform EM iterations
    for i in range(maxiter):
        # E-step: Calculate expected values of heads and tails for each coin
        heads_A, tails_A, heads_B, tails_B = e_step(rolls, theta_A, theta_B)
        # M-step: Update theta_A and theta_B based on the expected values
        theta_A, theta_B = m_step(heads_A, tails_A, heads_B, tails_B)

    thetas.append((theta_A, theta_B))  # Add the final theta values to the list
    return thetas, (theta_A, theta_B)


# E-step: Calculate expected values of heads and tails for each coin
def e_step(rolls, theta_A, theta_B):
    heads_A, tails_A = 0, 0
    heads_B, tails_B = 0, 0

    # Iterate over each trial
    for trial in rolls:
        # Calculate the likelihood of the trial for each coin
        likelihood_A = coin_likelihood(trial, theta_A)
        likelihood_B = coin_likelihood(trial, theta_B)

        # Calculate the probabilities of the trial belonging to coin A and coin B
        p_A = likelihood_A / (likelihood_A + likelihood_B)
        p_B = likelihood_B / (likelihood_A + likelihood_B)

        # Update the expected values of heads and tails for each coin
        heads_A += p_A * trial.count("H")
        tails_A += p_A * trial.count("T")
        heads_B += p_B * trial.count("H")
        tails_B += p_B * trial.count("T")

    return heads_A, tails_A, heads_B, tails_B


# M-step: Update theta_A and theta_B based on the expected values
def m_step(heads_A, tails_A, heads_B, tails_B):
    theta_A = heads_A / (heads_A + tails_A)
    theta_B = heads_B / (heads_B + tails_B)
    return theta_A, theta_B


# Calculate the likelihood of a coin toss sequence given a bias
def coin_likelihood(roll, bias):
    numHeads = roll.count("H")
    flips = len(roll)
    return pow(bias, numHeads) * pow(1-bias, flips-numHeads)


# Coin toss sequences and initial theta values
# x = [ "HTTTHHTHTH", "HHHHTHHHHH", "HTHHHHHTHH", "HTHTTTHHTT", "THHHTHHHTH" ] //example
x = ['HHHHHTTTTTHHH', 'HHTHHTHTHTHHT', 'HHTHTTHTHTHTT', 'TTTHTTHTHTHTT', 'THTHTTHTHTHHH']
thetas, _ = coin_em(rolls=x, theta_A=0.6, theta_B=0.5, maxiter=6)


# Function to plot the likelihood of coin biases
def plot_coin_likelihood(rolls, thetas=None, figout='Documents/base.png'):
    xvals = np.linspace(0.01, 0.99, 100)
    yvals = np.linspace(0.01, 0.99, 100)
    X, Y = np.meshgrid(xvals, yvals)

    Z = []
    # Calculate the likelihood for each combination of theta_A and theta_B
    for i, r in enumerate(X):
        z = []
        for j, c in enumerate(r):
            z.append(coin_marginal_likelihood(rolls, c, Y[i][j]))
        Z.append(z)

    plt.figure(figsize=(10, 8))
    # Create a contour plot of the likelihood values
    C = plt.contour(X, Y, Z, 150)
    cbar = plt.colorbar(C)
    plt.title(r"Likelihood $\log p(\mathcal{X}|\theta_A,\theta_B)$", fontsize=20)
    plt.xlabel(r"$\theta_A$", fontsize=20)
    plt.ylabel(r"$\theta_B$", fontsize=20)

    if thetas is not None:
        thetas = np.array(thetas)
        # Plot the trajectory of theta_A and theta_B during EM iterations
        plt.plot(thetas[:, 0], thetas[:, 1], '-k', lw=2.0)
        plt.plot(thetas[:, 0], thetas[:, 1], 'ok', ms=5.0)


# Calculate the marginal likelihood of coin toss sequences given biases
def coin_marginal_likelihood(rolls, biasA, biasB):
    trials = []
    for roll in rolls:
        h = roll.count("H")
        t = roll.count("T")
        likelihoodA = coin_likelihood(roll, biasA)
        likelihoodB = coin_likelihood(roll, biasB)
        trials.append(np.log(0.5 * (likelihoodA + likelihoodB)))
    return sum(trials)


# Plot the likelihood of coin biases
plot_coin_likelihood(x, thetas)
plt.savefig("./assets/coin-plot-likelihood.png")

print(x)  # Print the coin toss sequences
