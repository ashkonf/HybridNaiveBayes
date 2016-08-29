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
    
    def featurize(self, object):
        if self.featurizer is None:
            raise Exception("If no featurizer is provided upon initialization, self.featurize must be overridden.")
        return self.featurizer(object)

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
            # A label count can never be 0 because we only generate
            # a label count upon observing the first data point that
            # belongs to it. As a result, we don't worrying about
            # the argument to log being 0 here.
            self.priors[label] = math.log(labelCounts[label])

    def __labelWeights(self, object):
        features = self.featurize(object)

        labelWeights = copy.deepcopy(self.priors)

        for feature in features:
            for label in self.priors:
                if feature.name in self.distributions[label]:
                    distribution = self.distributions[label][feature.name]
                    if isinstance(distribution, distributions.DiscreteDistribution):
                        probability = distribution.probability(feature.value)
                    elif isinstance(distribution, distributions.ContinuousDistribution):
                        probability = distribution.pdf(feature.value)
                    else:
                        raise Exception("Naive Bayes Training Error: Invalid probability distribution")
            
                else:
                    if issubclass(feature.distribution, distributions.Binary):
                        distribution = distributions.Binary(0, self.priors[label])
                        probability = distribution.probability(feature.value)
                    else:
                        raise Exception("Naive Bayes Training Error: Non-binary features must be present for all training examples")

                if probability == 0.0: labelWeights[label] = float("-inf")
                else: labelWeights[label] += math.log(probability)

        return labelWeights
    
    def probability(self, object, label):
        labelWeights = self.__labelWeights(object)
        
        numerator = labelWeights[label]
        if numerator == float("-inf"): return 0.0
        
        denominator = 0.0
        minWeight = min(labelWeights.iteritems(), key=operator.itemgetter(1))[1]
        for label in labelWeights:
            weight = labelWeights[label]
            if minWeight < 0.0: weight /= (-minWeight)
            denominator += math.exp(weight)
        denominator = math.log(denominator)
        
        return math.exp(numerator - denominator)

    def probabilities(self, object):
        labelProbabilities = collections.Counter()
        for label in self.priors:
            labelProbabilities[label] = self.probability(object, label)
        return labelProbabilities

    def classify(self, object, costMatrix=None):
        if costMatrix is None:
            labelWeights = self.__labelWeights(object)
            return max(labelWeights.iteritems(), key=operator.itemgetter(1))[0]
        
        else:
            labelCosts = collections.Counter()
            labelProbabilities = self.probabilities(object)
            for predictedLabel in labelProbabilities:
                if predictedLabel not in costMatrix: raise Exception("Naive Bayes Prediction Error: Cost matrix does not include all labels.")
                cost = 0.0
                for actualLabel in labelProbabilities:
                    if actualLabel not in costMatrix: raise Exception("Naive Bayes Prediction Error: Cost matrix does not include all labels.")
                    cost += labelProbabilities[predictedLabel] * costMatrix[predictedLabel][actualLabel]
                labelCosts[predictedLabel] = cost
            return min(labelCosts.iteritems(), key=operator.itemgetter(1))[0]

    def accuracy(self, objects, goldLabels):
        if len(objects) == 0 or len(objects) != len(goldLabels):
            raise ValueError("Malformed data")
        
        numCorrect = 0
        for index, object in enumerate(objects):
            if self.classify(object) == goldLabels[index]:
                numCorrect += 1
        return float(numCorrect) / float(len(objects))
