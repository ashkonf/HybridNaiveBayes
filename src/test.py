import requests
import nb
import distributions

# The data set is described here: https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)
raw_data = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data").text.strip()

lines = raw_data.split("\n")
value_matrix = [line.split() for line in lines]
data_points = [values[:-1] for values in value_matrix]
labels = [values[-1] for values in value_matrix]

data_set_slice = len(data_points) / 2
training_set = (data_points[:data_set_slice], labels[:data_set_slice])
test_set = (data_points[data_set_slice:], labels[data_set_slice:])

feature_distributions = [
    distributions.Multinomial, # Checking account status is bucketed and therefore categorical
    distributions.Exponential, # Duration in months is continuous and probably follows a power law distribution
    distributions.Multinomial, # Credit history is categorical
    distributions.Multinomial, # Purpose is categorical
    distributions.Gaussian, # Credit amount is continuous and probably follows a normal distribution
    distributions.Multinomial, # Savings account status is bucketed and therefore categorical
    distributions.Multinomial, # Unemployment duration is bucketed and therefore categorical
    distributions.Gaussian, # Installment rate is continuous and probably follows a normal distribution
    distributions.Multinomial, # Personal status is categorical
    distributions.Multinomial, # Other debtors is categorical
    distributions.Exponential, # Present residence since is continuous and probably follows a power law distribution
    distributions.Multinomial, # Property status is categorical
    distributions.Gaussian, # Age is continuous and probably follows a normal distribution
    distributions.Multinomial, # Other installment plans is categorical
    distributions.Multinomial, # Housing is categorical
    distributions.Exponential, # Number of credit cards is continuous and probably follows a power law distribution
    distributions.Multinomial, # Job is categorical
    distributions.Exponential, # Number of people liable continuous and probably follows a power law distribution
    distributions.Multinomial, # Telephone is categorical
    distributions.Multinomial, # Foreign worker is categorical
]

def featurizer(data_point):
    features = []
    for index, value in enumerate(data_point):
        name = str(index)
        distribution = feature_distributions[index]
        try: value = float(value)
        except: pass
        features.append(nb.Feature(name, distribution, value))
    return features

classifier = nb.NaiveBayesClassifier(featurizer)
classifier.train(training_set[0], training_set[1])
print "Accuracy = %s" % classifier.accuracy(test_set[0], test_set[1])
