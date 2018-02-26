# Hybrid Naive Bayes


A generalized implementation of the Naive Bayes classifier in Python that provides the following functionality:

1. Support for both categorical and ordered features.
2. Support for both discrete and continuous ordered features.
3. Support for modeling ordered features using arbitrary probability distributions.

## Setup

There's not much to it; just include the *nb.py* and *distributions.py* files in your project, and use away!

## Details

Most Naive Bayes implementations allow only for categorical features. A small number of implementations, such as scikit-learn's implementation of the Gaussian Naive Bayes classifier, allow for ordered features. These implementations sacrifice functionality to do so. For example, scikit-learn's implementation of the Gaussian Naive Bayes classifier sacrifices support for categorical features and models every feature using the Normal Distribution. Note that because it models all ordered features using the Normal Distribution, it only supports continuous ordered features and provides no support for discrete ordered features.

This Naive Bayes Classifier implementation has none of the above mentioned limitations. First and foremost, it allows for the simultaneous use of categorical and ordered features. In addition and in some cases just as importantly, it allows the client to specify the distributions used to model ordered features. There is no assumption that every ordered feature must conform to the Gaussian Distribution. This library provides implementations of many commonly used probability distributions the client can elect to use in modeling ordered features. Furthermore, the implementation of the Naive Bayes Classifier allows the client to subclass the DiscreteDistribution or ContinuousDistribution classes to implement any distribution this library may not have implemented for use in modeling ordered features. In short, this library allows clients to model ordered features using any valid probability distribution imaginable.

See Derivation.pdf for a full mathematical justification of this implementation of the Naive Bayes Classifier.

## Sample Usage

**TL;DR**: Run *test.py*.

**More Details**: For an sample usage of this Naive Bayes classifier implementation, see *test.py*. It demonstrates how to use the classifier by downloading a credit-related data set hosted by UCI, training the classifier on half the data in the data set, and evaluating the classifier's performance on the other half. One can access the data [here](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data), and read more about its composition [here](https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)).
