import numpy as np
import scipy.stats as stats
import scipy.special as spc
import matplotlib.pyplot as plt

class StockDaySim():
    """Simulate a stock price quote as a 1D-random walker at time t.
        This walker is defined as the Wiener Process W(t)=Nc*B(t).
        B(t) is a Brownian Bridge process. Nc is an independent normal
        distribution for scaling.

        For t on the interval [0,1], the walker travels stochastically
        from W(0)=x to W(1)=y, achieving the values w and z as the min
        and max values observed with sample size n. 

        Broadcasting works (numpy vectorized).
        """

    def __init__(self, dayopen, dayclose, low, high, n):
        self.low = low
        self.high = high
        self.open = dayopen
        self.close = dayclose
        self.n = n  # should be similar to the sample size observed by the neural network
        self.x = (dayopen - low)/(high - low)
        self.y = (dayclose - low)/(high - low)
        self.times = []
        self.dist = []
        
    def generate(self, t):
        """Generate the normal distribution representing a random variable of
        the stock price at time t.
        
        The distribution is truncated at the ends of the time interval.
        """
        raw_mu, raw_std = multiply_norms(
            *brownian_bridge(t, self.x, self.y), *brownian_multiplier(1, self.n))
        
        ## Scale and relocate
        mu = raw_mu * (self.high - self.low) + self.low
        std = raw_std * (self.high - self.low)

        return stats.truncnorm((self.low - mu) / std, (self.high - mu) / std, loc=mu, scale=std)
    
    def sample(self, t, count=None):
        """Sample the distribution(s) at time t."""
        X = self.generate(t)
        if count:
            count = (count, self.n)
        return X.rvs(count).T

    def simulate_many(self, times=None):
        """Creates simulations for all of the days at the given times."""
        if not times:
            times = [0.01, 0.1, 0.3, 0.5, 0.8, 0.9, 0.99, 0.999]
        self.times = times
        self.dist = []
        for t in times:
            self.dist.append(self.sample(t, 10000))
    
    def plot_day_sim(self, day):
        """Plots the simulation at various times for a given historical day.
        
        Days are specified as number of days in the past (e.g. -7).
        """
        if not self.dist:
            self.simulate_many()
        fig, ax = plt.subplots(len(self.times), sharex=True)
        for i, t in enumerate(self.times):
            ax[i].hist(self.dist[i][day], density=True, bins=40)
        plt.show()

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
