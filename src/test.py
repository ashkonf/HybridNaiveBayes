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

def featurizer(data_point):
    return [
        nb.Feature("Checking account status", distributions.Multinomial, data_point[0]), # bucketed and therefore categorical
        nb.Feature("Duration in months", distributions.Exponential, float(data_point[1])), # continuous and probably follows a power law distribution
        nb.Feature("Credit history", distributions.Multinomial, data_point[2]), # categorical
        nb.Feature("Purpose", distributions.Multinomial, data_point[3]), # categorical
        nb.Feature("Credit amount", distributions.Gaussian, float(data_point[4])), # continuous and probably follows a normal distribution
        nb.Feature("Savings account status", distributions.Multinomial, data_point[5]), # bucketed and therefore categorical
        nb.Feature("Unemployment duration", distributions.Multinomial, data_point[6]), # bucketed and therefore categorical
        nb.Feature("Installment rate", distributions.Gaussian, float(data_point[7])), # continuous and probably follows a normal distribution
        nb.Feature("Personal status", distributions.Multinomial, data_point[8]), # categorical
        nb.Feature("Other debtors", distributions.Multinomial, data_point[9]), # categorical
        nb.Feature("Present residence", distributions.Exponential, float(data_point[10])), # continuous and probably follows a power law distribution
        nb.Feature("Property status", distributions.Multinomial, data_point[11]), # categorical
        nb.Feature("Age", distributions.Gaussian, float(data_point[12])), # continuous and probably follows a normal distribution
        nb.Feature("Other installment plans", distributions.Multinomial, data_point[13]), # categorical
        nb.Feature("Housing", distributions.Multinomial, data_point[14]), # categorical
        nb.Feature("Number of credit cards", distributions.Exponential, float(data_point[15])), # continuous and probably follows a power law distribution
        nb.Feature("Job", distributions.Multinomial, data_point[16]), # categorical
        nb.Feature("Number of people liable", distributions.Exponential, float(data_point[17])), # continuous and probably follows a power law distribution
        nb.Feature("Telephone", distributions.Multinomial, data_point[18]), # categorical
        nb.Feature("Foreign worker", distributions.Multinomial, data_point[19]) # categorical
    ]

classifier = nb.NaiveBayesClassifier(featurizer)
classifier.train(training_set[0], training_set[1])
print "Accuracy = %s" % classifier.accuracy(test_set[0], test_set[1])
