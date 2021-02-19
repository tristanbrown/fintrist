import numpy as np
import scipy.stats as stats
import scipy.special as spc
import matplotlib.pyplot as plt

class StockDaySim():
    def __init__(self, x, y, w, z, n):
        self.x = x
        self.y = y
        self.w = w
        self.z = z
        self.n = n
        
    def generate_stocksim(self, t):
        """Generate a normal distribution that represents a 1D-random
        walker at time t.
        This walker is defined as the Wiener Process W(t)=Nc*B(t).
        B(t) is a Brownian Bridge process. Nc is an independent normal
        distribution for scaling.

        For t on the interval [0,1], the walker travels stochastically
        from W(0)=x to W(1)=y, achieving the values w and z as the min
        and max values observed with sample size n. 

        The distribution is truncated at the ends of the time interval.
        """
        mu, std = multiply_norms(
            *brownian_bridge(t, self.x, self.y), *brownian_multiplier(self.z, self.n))
        return stats.truncnorm((0 - mu) / std, (1 - mu) / std, loc=mu, scale=std)

def brownian_bridge(t, x, y):
    """Give the mean and std dev of a normal distribution
    representing a Brownian Bridge B(t) tethered at
    B(0)=x, B(1)=y
    """
    mu = x*(1 - t) + y*t
    var = t*(1 - t)
    std = np.sqrt(var)
    return mu, std

def norm_quantile(p):
    """Give the quantile (number of std devs) on each side
    of the mean containing the given cumulative probability.
    """
    return np.sqrt(2)*spc.erfinv(2*p-1)

def norm_limited(w, z, n):
    """Give the mean and std dev of the normal distribution
    from which w and z are expected to be the observed min
    and max, at a given sample size n. 

    p is the cumulative probability containing every point
        except the min and max for a given sample size n.
    """
    p = (n-1)/n
    mu = (w + z)/2
    std = (z - w)/(2 * norm_quantile(p))
    return mu, std

def brownian_multiplier(T, n):
    """Give the mean and std dev of an independent,
    time-invariant normal distribution that scales the
    a Brownian Bridge such that its time-integral spans
    the range [0,T], for a given sample size n.
    """
    muT, stdT = norm_limited(0, T, n)
    varC = (stdT**2)/(1 - 3/(T**3)*(stdT**2))
    muC = muT * (3/(T**3)*varC + 1)
    stdC = np.sqrt(varC)
    return muC, stdC

def multiply_norms(mu1, std1, mu2, std2):
    """Give the mean and std dev of a normal distribution
    representing the joint probability product of two
    independent normal distributions.
    """
    mu = (mu1*(std2**2) + mu2*(std1**2))/(std1**2 + std2**2)
    var = (std1**2)*(std2**2)/(std1**2 + std2**2)
    std = np.sqrt(var)
    return mu, std

def plot_stock_sim():
    times = [0.01, 0.1, 0.3, 0.5, 0.8, 0.9, 0.99, 0.999]
    x, y = 0.7, 0.9
    w, z = 0, 1
    # n = 60 * 6.5
    n=7000  # should be similar to the sample size observed by the neural network
    # n=55000000
    sim=StockDaySim(x, y, w, z, n)

    fig, ax = plt.subplots(len(times)+1, sharex=True)
    X = stats.norm(*norm_limited(w, z, n))
    ax[0].hist(X.rvs(10000), density=True, bins=40)
    for i, t in enumerate(times):
        X = sim.generate_stocksim(t)
        ax[i+1].hist(X.rvs(10000), density=True, bins=40)
    # ax[1].hist(N.rvs(10000), density=True)
    plt.xlim([0,1])
    plt.show()
