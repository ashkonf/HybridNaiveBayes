import os
import sys
import collections
import math
import operator
import copy

import distributions

## Feature ##############################################################################

class Feature(object):

    def __init__(self, name, distribution, value):
        self.name = name
        self.distribution = distribution
        self.value = value

    def __repr__(self):
        return self.name + " => " + str(self.value)
    
    def hashable(self):
        return (self.name, self.value)

    @classmethod
    def binary(cls, name):
        return cls(name, distributions.Binary, True)

## ExtractedFeature #####################################################################

class ExtractedFeature(Feature):

    def __init__(self, object):
        name = self.__class__.__name__
        distribution = self.distribution()
        value = self.extract(object)
        super(ExtractedFeature, self).__init__(name, distribution, value)

    def extract(self, object):
        # returns feature value corresponding to |object|
        raise NotImplementedError("Subclasses should override.")

    @classmethod
    def distribution(cls):
        # returns the distribution this feature conforms to
        raise NotImplementedError("Subclasses should override.")

## NaiveBayesClassifier #################################################################

class NaiveBayesClassifier(object):

    def __init__(self, featurizer = None):
        self.featurizer = featurizer
        self.priors = None
        self.distributions = None

    def train(self, objects, labels):
        featureValues = collections.defaultdict(lambda: collections.defaultdict(lambda: []))
        distributionTypes = {}

        labelCounts = collections.Counter()

        for index, object in enumerate(objects):
            label = labels[index]
            labelCounts[label] += 1
            for feature in self.featurize(object):
                featureValues[label][feature.name].append(feature.value)
                distributionTypes[feature.name] = feature.distribution

        self.distributions = collections.defaultdict(lambda: {})
        for label in featureValues:
            for featureName in featureValues[label]:
                try:
                    values = featureValues[label][featureName]
                    if issubclass(distributionTypes[featureName], distributions.Binary):
                        trueCount = len([value for value in values if value])
                        # the absence of binary feature is treated as it having been present with a False value
                        falseCount = labelCounts[label] - trueCount
                        distribution = distributions.Binary(trueCount, falseCount)
                    else:
                        distribution = distributionTypes[featureName].mleEstimate(values)
                except distributions.EstimationError, distributions.ParametrizationError:
                    if issubclass(distributionTypes[featureName], distributions.Binary):
                        distribution = distributions.Binary(0, labelCounts[label])
                    elif issubclass(distributionTypes[featureName], distributions.DiscreteDistribution):
                        distribution = distributions.DiscreteUniform(-sys.maxint, sys.maxint)
                    else:
                        distribution = distributions.Uniform(-sys.float_info.max, sys.float_info.max)
                self.distributions[label][featureName] = distribution

        self.priors = collections.Counter()
        for label in labelCounts:
            self.priors[label] = math.log(labelCounts[label])

    def __labelWeights(self, object):
        features = self.featurize(object)

        labelWeights = copy.deepcopy(self.priors)

        for feature in features:
            for label in self.labelCounts:
                if feature.name in self.distributions[label]:
                    distribution = self.distributions[label][feature.name]
                    if isinstance(distribution, distributions.DiscreteDistribution):
                        labelWeights[label] += math.log(distribution.probability(feature.value))
                    elif isinstance(distribution, distributions.ContinuousDistribution):
                        labelWeights[label] += math.log(distribution.pdf(feature.value))
                    else:
                        raise Exception("invalid probability distribution")
                else:
                    if issubclass(feature.distribution, distributions.Binary):
                        distribution = distributions.Binary(0, self.labelCounts[label])
                        labelWeights[label] += math.log(distribution.probability(True))
                    else:
                        raise Exception("non-binary features must be present for all training examples")

        return labelWeights

    def classify(self, object):
        labelWeights = self.__labelWeights(object)
        return max(labelWeights.iteritems(), key=operator.itemgetter(1))[0]

    def probability(self, object, label):
        labelWeights = self.__labelWeights(object)
        numerator = labelWeights[label]
        denominator = 0.0
        for label in labelWeights:
            denominator += math.exp(labelWeights[label])
        denominator = math.log(denominator)
        return math.exp(numerator - denominator)

    def featurize(self, object):
        if self.featurizer is None:
            raise Exception("If no featurizer is provided upon initialization, self.featurize must be overridden.")
        return self.featurizer(object)
