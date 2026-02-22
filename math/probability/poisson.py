#!/usr/bin/env python3
"""Poisson distribution class"""


class Poisson:
    """Represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize Poisson distribution

        Args:
            data: list of data to estimate the distribution
            lambtha: expected number of occurrences in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Calculate lambtha as the mean of the data
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculate the PMF for a given number of successes

        Args:
            k: number of successes

        Returns:
            PMF value for k
        """
        k = int(k)
        if k < 0:
            return 0

        # PMF formula: P(X = k) = (λ^k * e^(-λ)) / k!
        e = 2.7182818285
        
        # Calculate k! (factorial)
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        pmf_value = (self.lambtha ** k * e ** (-self.lambtha)) / factorial
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

        # CDF is the sum of PMF values from 0 to k
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)

        return cdf_value
