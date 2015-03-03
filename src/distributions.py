import os
import sys
import math
import collections

## Distribution ########################################################################################

class Distribution(object):

    def __init__(self):
        raise NotImplementedError("Not implemented")

    @classmethod
    def mleEstimate(cls, points):
        raise NotImplementedError("Not implemented")

    @classmethod
    def momEstimate(cls, points):
        raise NotImplementedError("Not implemented")

## ContinuousDistribution ##############################################################################

class ContinuousDistribution(Distribution):

    def pdf(self, value):
        raise NotImplementedError("Not implemented")

    def cdf(self, value):
        raise NotImplementedError("Not implemented")

## Uniform #############################################################################################

class Uniform(ContinuousDistribution):

    def __init__(self, alpha, beta):
        if alpha == beta: raise Exception("alpha and beta cannot be equivalent")
        self.alpha = alpha
        self.beta = beta
        self.range = beta - alpha

    def pdf(self, value):
        if value < self.alpha or value > self.beta: return 0.0
        else: return 1.0 / self.range

    def cdf(self, value):
        if value < self.alpha: return 0.0
        elif value >= self.beta: return 1.0
        else: return (value - self.alpha) / self.range

    def __str__(self):
        return "Continuous Uniform distribution: alpha = %s, beta = %s" % (self.alpha, self.beta)

    @classmethod
    def mleEstimate(cls, points):
        return cls(min(points), max(points))

## Gaussian ############################################################################################

class Gaussian(ContinuousDistribution):

    def __init__(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev
        if stdev == 0.0: raise Exception("standard deviation must be non-zero")
        if stdev < 0.0: raise Exception("standard deviation must be positive")
        self.variance = math.pow(stdev, 2.0)

    def pdf(self, value):
        numerator = math.exp(-math.pow(float(value - self.mean) / self.stdev, 2.0) / 2.0)
        denominator = math.sqrt(2 * math.pi * self.variance)
        return numerator / denominator

    def cdf(self, value):
        return 0.5 * (1.0 + math.erf((value - self.mean) / math.sqrt(2.0 * self.variance)))

    def __str__(self):
        return "Continuous Gaussian (Normal) distribution: mean = %s, standard deviation = %s" %(self.mean, self.stdev)

    @classmethod
    def mleEstimate(cls, points):
        numPoints = float(len(points))
        if numPoints <= 1: raise Exception("must provide at least 2 training points")

        mean = sum(points) / numPoints

        variance = 0.0
        for point in points:
            variance += math.pow(float(point) - mean, 2.0)
            variance /= (numPoints - 1.0)
        stdev = math.sqrt(variance)

        return cls(mean, stdev)

## Gaussian ############################################################################################

class BoundedGaussian(ContinuousDistribution):

    def __init__(self, mean, stdev, alpha, beta):
        self.mean = mean
        if stdev == 0.0: raise Exception("standard deviation must be non-zero")
        if stdev < 0.0: raise Exception("standard deviation must be positive")
        self.stdev = stdev
        self.variance = math.pow(stdev, 2.0)
        self.alpha = alpha
        self.beta = beta

    def pdf(self, value):
        if self.alpha == self.beta or self.__phi(self.alpha) == self.__phi(self.beta):
            if value == self.alpha: return 1.0
            else: return 0.0
        else:
            numerator = math.exp(-math.pow((value - self.mean) / self.stdev, 2.0) / 2.0)
            denominator = math.sqrt(2 * math.pi) * self.stdev * (self.__phi(self.beta) - self.__phi(self.alpha))
            return numerator / denominator

    def cdf(self, value):
        if value < self.alpha or value > self.beta:
            return 0.0
        else:
            numerator = self.__phi((value - self.mean) / self.stdev) - self.__phi(self.alpha)
            denominator = self.__phi(self.beta) - self.__phi(self.alpha)
            return numerator / denominator

    def __str__(self):
        return "Continuous Bounded Gaussian (Normal) distribution: mean = %s, standard deviation = %s, lower bound = %s, upper bound = %s" % (self.mean, self.stdev, self.alpha, self.beta)

    @classmethod
    def mleEstimate(cls, points):
        numPoints = float(len(points))

        if numPoints <= 1: raise Exception("must provide at least 2 training points")

        mean = sum(points) / numPoints

        variance = 0.0
        for point in points:
            variance += math.pow(float(point) - mean, 2.0)
        variance /= (numPoints - 1.0)
        stdev = math.sqrt(variance)

        return cls(mean, stdev, min(points), max(points))

    def __phi(self, value):
        return 0.5 * (1.0 + math.erf((value - self.mean) / (self.stdev * math.sqrt(2.0))))

## LogNormal ###########################################################################################

class LogNormal(ContinuousDistribution):

    def __init__(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev
        if stdev == 0.0: raise Exception("standard deviation must be non-zero")
        if stdev < 0.0: raise Exception("standard deviation must be positive")
        self.variance = math.pow(stdev, 2.0)

    def pdf(self, value):
        if value <= 0:
            return 0.0
        else:
            return math.exp(-math.pow(float(math.log(value) - self.mean) / self.stdev, 2.0) / 2.0) / (value * math.sqrt(2 * math.pi * self.variance))

    def cdf(self, value):
        return 0.5 + 0.5 * math.erf((math.log(value) - self.mean) / math.sqrt(2.0 * self.variance))

    def __str__(self):
        return "Continuous Log Normal distribution: mean = %s, standard deviation = %s" % (self.mean, self.stdev)

    @classmethod
    def mleEstimate(cls, points):
        numPoints = float(len(points))

        if numPoints <= 1: raise Exception("must provide at least 2 training points")

        mean = sum(math.log(float(point)) for point in points) / numPoints

        variance = 0.0
        for point in points:
            variance += math.pow(math.log(float(point)) - mean, 2.0)
        variance /= (numPoints - 1.0)
        stdev = math.sqrt(variance)

        return cls(mean, stdev)

## Exponential ########################################################################################

class Exponential(ContinuousDistribution):

    def __init__(self, lambdaa):
        # 2 "a"s to avoid confusion with "lambda" keyword
        self.lambdaa = lambdaa

    def pdf(self, value):
        return self.lambdaa * math.exp(-self.lambdaa * value)

    def cdf(self, value):
        return 1.0 - math.exp(-self.lambdaa * value)

    def __str__(self):
        return "Continuous Exponential distribution: lamda = %s" % self.lambdaa

    @classmethod
    def mleEstimate(cls, points):
        mean = float(sum(points))
        numPoints = len(points)
        if numPoints > 0:
            mean /= float(numPoints)
        lambdaa = 1.0 / mean if mean != 0 else 0.0

        return cls(lambdaa)

## Pareto #############################################################################################

class Pareto(ContinuousDistribution):

    def __init__(self, min, alpha):
        self.min = min
        self.alpha = alpha

    def pdf(self, value):
        if value < self.min: return 0.0
        else: return self.alpha * math.pow(self.min, self.alpha) / math.pow(value, self.alpha + 1.0)

    def cdf(self, value):
        if value < self.min: return 0.0
        else: return 1.0 - math.pow(self.min / value, self.alpha)

    def __str__(self):
        return "Continuous Pareto distribution: minimum = %s, alpha = %s" % (self.min, self.alpha)

    @classmethod
    def mleEstimate(cls, points):
        raise NotImplementedError("not implemented")

## Gamma #############################################################################################

class Gamma(ContinuousDistribution):

    def __init__(self, k, theta):
        self.k = k
        self.theta = theta
        self.denominator = math.gamma(self.k) * math.pow(self.theta, self.k)

    def pdf(self, value):
        return math.pow(value, self.k - 1.0) * math.exp(-value / self.theta) / self.denominator

    def cdf(self, value):
        raise NotImplementedError("not implemented")

    def __str__(self):
        return "Continuous Gamma distribution: k = %s, theta = %s" % (self.k, self.theta)

    @classmethod
    def mleEstimate(cls, points):
        raise NotImplementedError("not implemented")

## Beta ##############################################################################################

class Beta(ContinuousDistribution):

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def pdf(self, value):
        return math.pow(value, self.alpha - 1.0) * math.pow(1.0 - value, self.beta - 1.0)

    def cdf(self, value):
        raise NotImplementedError("not implemented")

    def __str__(self):
        return "Continuous Beta distribution: alpha = %s, beta = %s" % (self.alpha, self.beta)

    @classmethod
    def mleEstimate(cls, points):
        raise NotImplementedError("not implemented")

## KernelDensityEstimate ##############################################################################

class KernelDensityEstimate(ContinuousDistribution):

    def __init__(self, observedPoints):
        self.observedPoints = observedPoints
        self.numObservedPoints = float(len(observedPoints))

    def pdf(self, value):
        stdev = 1.0 / math.sqrt(self.numObservedPoints)
        sum = 0.0
        for point in self.observedPoints:
            sum += self.normalPdf(point, stdev, value)
        return sum / self.numObservedPoints

    def normalPdf(self, mean, stdev, value):
        variance = math.pow(stdev, 2.0)
        numerator = math.exp(-math.pow(float(value - mean) / stdev, 2.0) / 2.0)
        denominator = math.sqrt(2 * math.pi * variance)
        return numerator / denominator

    def cdf(self, value):
        raise NotImplementedError("Not implemented")

    def __str__(self):
        return "Continuous Gaussian Kernel Density Estimate distribution"

    @classmethod
    def mleEstimate(cls, points):
        return cls(points)

## DiscreteDistribution ###############################################################################

class DiscreteDistribution(Distribution):

    def probability(self, value):
        raise NotImplementedError("Not implemented")

## Uniform ############################################################################################

class DiscreteUniform(DiscreteDistribution):

    def __init__(self, alpha, beta):
        if alpha == beta: raise Exception("alpha and beta cannot be equivalent")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.prob = 1.0 / (self.beta - self.alpha)

    def probability(self, value):
        if value < self.alpha or value > self.beta: return 0.0
        else: return self.prob

    def __str__(self):
        return "Discrete Uniform distribution: alpha = %s, beta = %s" % (self.alpha, self.beta)

    @classmethod
    def mleEstimate(cls, points):
        return cls(min(points), max(points))

## Poissoin ###########################################################################################

class Poisson(DiscreteDistribution):

    def __init__(self, lambdaa):
        # 2 "a"s to avoid confusion with "lambda" keyword
        self.lambdaa = lambdaa

    def probability(self, value):
        first = float(math.pow(self.lambdaa, value)) / float(math.factorial(value))
        second = math.exp(-float(self.lambdaa))
        return first * second

    def __str__(self):
        return "Discrete Poisson distribution: lamda = %s" % self.lambdaa

    @classmethod
    def mleEstimate(cls, points):
        mean = float(sum(points)) / float(len(points))
        return cls(mean)

## Multinomial #######################################################################################

class Multinomial(DiscreteDistribution):

    def __init__(self, categoryCounts, smoothingFactor = 1.0):
        self.categoryCounts = categoryCounts
        self.numPoints = float(sum(categoryCounts.values()))
        self.numCategories = float(len(categoryCounts))
        self.smoothingFactor = float(smoothingFactor)

    def probability(self, value):
        if not value in self.categoryCounts:
            return 0.0
        numerator = float(self.categoryCounts[value]) + self.smoothingFactor
        denominator = self.numPoints + self.numCategories * self.smoothingFactor
        return numerator / denominator

    def __str__(self):
        return "Discrete Multinomial distribution: buckets = %s" % self.categoryCounts

    @classmethod
    def mleEstimate(cls, points):
        categoryCounts = collections.Counter()
        for point in points:
            categoryCounts[point] += 1
        return cls(categoryCounts)

## Binary ############################################################################################

class Binary(Multinomial):

    def __init__(self, trueCount, falseCount, smoothingFactor = 1.0):
        categoryCounts = { True : trueCount, False : falseCount }
        Multinomial.__init__(self, categoryCounts, smoothingFactor)

    def __str__(self):
        return "Discrete Binary distribution: true count = %s, false count = %s" % (self.categoryCounts[True], self.categoryCounts[False])

    @classmethod
    def mleEstimate(cls, points, smoothingFactor = 1.0):
        trueCount = 0
        for point in points:
            if point: trueCount += 1
        falseCount = len(points) - trueCount
        return cls(trueCount, falseCount, smoothingFactor)

## Errors ############################################################################################

class EstimationError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class ParametrizationError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
