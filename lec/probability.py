import numpy as np
import matplotlib.pyplot as plt

def flip_a_coin(p=0.5, print_stuff=False):
    """
    Simulates flipping a biased coin.

    Parameters:
    ----------
    p : float, optional
        The probability of getting a head. Default is 0.5 (fair coin).
    print_stuff : bool, optional
        If True, prints "Head" or "Tail" after each flip. Default is False.

    Returns:
    -------
    int
        Returns 1 if the result is heads, 0 if tails.
    """
    draw = np.random.uniform(0, 1)
    if draw >= p:
        if print_stuff:
            print("Head")
        return 1
    else:
        if print_stuff:
            print("Tail")
        return 0

def flip_many_times(n=20):
    """
    Flips a coin `n` times and calculates the proportion of heads.

    Parameters:
    ----------
    n : int, optional
        The number of times to flip the coin. Default is 20.

    Returns:
    -------
    float
        The proportion of heads (number of heads / number of flips).
    """
    num_heads = 0
    for i in range(n):
        num_heads += flip_a_coin()

    return num_heads / n

def plot_coin_flip(start, stop, by):
    """
    Plots the proportion of heads over a range of coin flips.

    Parameters:
    ----------
    start : int
        The starting number of flips.
    stop : int
        The stopping number of flips (exclusive).
    by : int
        The step size between consecutive numbers of flips.
    """
    times = np.arange(start, stop, step=by)
    results = [flip_many_times(k) for k in times]

    plt.figure()
    plt.scatter(times, results)
    plt.hlines(0.5, 0, 2000, linestyles="dashed")
    plt.xlabel('Number of Coin Flips')
    plt.ylabel('Proportion of Heads')
    plt.title('Proportion of Heads as Coin Flips Increase')
    plt.show()

plot_coin_flip(20, 2000, 10)

def flip_variance(n: int):
    """
    Calculates the variance of heads after `n` flips.

    Parameters:
    ----------
    n : int
        The number of coin flips.

    Returns:
    -------
    float
        The variance of the coin flips' outcome.
    """
    flips = [flip_a_coin() for i in range(n)]
    mu = np.mean(flips)
    var = np.sum((flips - mu) ** 2) / n

    return var

print("Variance of number of heads after 20 flips is: ", flip_variance(20), ".")

def plot_flip_variance(start, stop, by):
    """
    Plots the variance of heads over a range of coin flips.

    Parameters:
    ----------
    start : int
        The starting number of flips.
    stop : int
        The stopping number of flips (exclusive).
    by : int
        The step size between consecutive numbers of flips.
    """
    times = np.arange(start, stop, step=by)
    vars = [flip_variance(k) for k in times]

    plt.figure()
    plt.scatter(times, vars)
    plt.hlines(y=0.25, xmin=0, xmax=2000, linestyles="dashed")
    plt.xlabel('Number of Coin Flips')
    plt.ylabel('Variance of Heads')
    plt.title('Variance of Heads as Coin Flips Increase')
    plt.show()

plot_flip_variance(20, 2000, 10)




# Plot exponential distrbution
def plot_exponential(l=1):
    disc = np.arange(0, 10, step=0.1)
    exps = [l * np.exp(-l * x) for x in disc]
    plt.figure()
    plt.scatter(disc, exps)
    plt.show()

# plot with various lambdas
plot_exponential(0.1)
plot_exponential(0.5)
plot_exponential(1)
plot_exponential(3)

# Check the mean and variance of exponential distribution
def sample_exponential(l, n):
    return [np.random.exponential(1/l) for i in range(n)]

def exponential_mean_and_var(l, n):
    sample = sample_exponential(l, n)
    sample_mean = np.mean(sample)
    sample_variance = np.var(sample)

    return sample_mean, sample_variance

def exponential_experiment(ls, n):
    mean_diff = []
    var_diff = []
    for l in ls:
        mean, var = exponential_mean_and_var(l, n)
        mean_diff.append(mean - (1 / l))
        var_diff.append(var - (1 / (l ** 2)))
        # if not (np.isclose(mean, 1 / l, atol=0.5) and np.isclose(var, 1 / (l ** 2), atol=0.5)):
        #     return False

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.scatter(ls, mean_diff)
    plt.xlabel("Lambda")
    plt.ylabel("Mean Error")
    plt.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(ls, var_diff)
    plt.xlabel("Lambda")
    plt.ylabel("Var Error")
    plt.show()

L = np.arange(0.05, 10, 0.05)
n = 5000

print(exponential_experiment(L, n))

