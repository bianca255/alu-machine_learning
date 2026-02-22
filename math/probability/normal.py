#!/usr/bin/env python3
"""Normal distribution class"""


class Normal:
    """Represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize Normal distribution

        Args:
            data: list of data to estimate the distribution
            mean: mean of the distribution
            stddev: standard deviation of the distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean
            self.mean = sum(data) / len(data)

            # Calculate standard deviation
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        Calculate the z-score of a given x-value

        Args:
            x: x-value

        Returns:
            z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculate the x-value of a given z-score

        Args:
            z: z-score

        Returns:
            x-value of z
        """
        return self.mean + z * self.stddev

    def pdf(self, x):
        """
        Calculate the PDF for a given x-value

        Args:
            x: x-value

        Returns:
            PDF value for x
        """
        # PDF formula: f(x) = (1 / (σ * √(2π))) * e^(-(x-μ)²/(2σ²))
        pi = 3.1415926536
        e = 2.7182818285

        coefficient = 1 / (self.stddev * (2 * pi) ** 0.5)
        exponent = -((x - self.mean) ** 2) / (2 * self.stddev ** 2)
        pdf_value = coefficient * e ** exponent

        return pdf_value

    def cdf(self, x):
        """
        Calculate the CDF for a given x-value

        Args:
            x: x-value

        Returns:
            CDF value for x
        """
        # CDF formula using error function approximation
        # CDF(x) = (1/2) * [1 + erf((x - μ) / (σ * √2))]

        pi = 3.1415926535897932

        # Calculate the value for the error function
        value = (x - self.mean) / (self.stddev * (2 ** 0.5))

        # Using a high-precision erf approximation
        # erf(x) ≈ (2/√π) * ∫₀ˣ e^(-t²) dt
        # Using numerical integration with Simpson's method

        # For the error function, we'll use a series expansion
        erf_value = self._erf(value)

        cdf_value = (1 + erf_value) / 2

        return cdf_value

    def _erf(self, x):
        """
        Calculate error function using series expansion

        Args:
            x: input value

        Returns:
            erf(x)
        """
        # Use the formula: erf(x) = (2/√π) * sum from n=0 to
        # infinity of ((-1)^n * x^(2n+1)) / (n! * (2n+1))

        pi = 3.1415926535897932
        e = 2.7182818284590452

        # For better accuracy, use Abramowitz and Stegun formula
        sign = 1 if x >= 0 else -1
        x = abs(x)

        # Constants
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        t = 1.0 / (1.0 + p * x)
        poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
        y = 1.0 - poly * (e ** (-x * x))

        return sign * y
