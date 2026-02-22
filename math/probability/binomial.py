#!/usr/bin/env python3
"""Binomial distribution class"""


class Binomial:
    """Represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize Binomial distribution

        Args:
            data: list of data to estimate the distribution
            n: number of Bernoulli trials
            p: probability of success
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean and variance
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # For binomial: mean = n*p, variance = n*p*(1-p)
            # From variance/mean = (1-p), we can find p
            # p = 1 - (variance / mean)
            p = 1 - (variance / mean)

            # From mean = n*p, we can find n
            # n = mean / p
            n = mean / p

            # Round n to nearest integer
            self.n = round(n)

            # Recalculate p with rounded n
            self.p = mean / self.n

    def pmf(self, k):
        """
        Calculate the PMF for a given number of successes

        Args:
            k: number of successes

        Returns:
            PMF value for k
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        # PMF formula: P(X = k) = C(n,k) * p^k * (1-p)^(n-k)
        # where C(n,k) = n! / (k! * (n-k)!)

        # Calculate C(n, k) - binomial coefficient
        n_factorial = 1
        for i in range(1, self.n + 1):
            n_factorial *= i

        k_factorial = 1
        for i in range(1, k + 1):
            k_factorial *= i

        nk_factorial = 1
        for i in range(1, self.n - k + 1):
            nk_factorial *= i

        binomial_coeff = n_factorial / (k_factorial * nk_factorial)

        pmf_value = binomial_coeff * (self.p ** k)
        pmf_value = pmf_value * ((1 - self.p) ** (self.n - k))
        return pmf_value

    def cdf(self, k):
        """
        Calculate the CDF for a given number of successes

        Args:
            k: number of successes

        Returns:
            CDF value for k
        """
        k = int(k)
        if k < 0:
            return 0
        if k > self.n:
            return 1

        # CDF is the sum of PMF values from 0 to k
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)

        return cdf_value
