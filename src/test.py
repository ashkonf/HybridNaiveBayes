import requests
import nb
import distributions

def load_data():
    # Loading, formatting and partitioning the data set:
    
    # This data set is described here: https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)
    raw_data = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data").text.strip()

    lines = raw_data.split("\n")
    value_matrix = [line.split() for line in lines]
    data_points = [values[:-1] for values in value_matrix]
    labels = [values[-1] for values in value_matrix]

    data_set_slice = len(data_points) / 2
    training_set = (data_points[:data_set_slice], labels[:data_set_slice])
    test_set = (data_points[data_set_slice:], labels[data_set_slice:])

    return (training_set, test_set)

def featurizer(data_point):
    # Massaging a data point into the format the NB classifier implementation expects of feature vectors:
    
    return [
        # Bucketed and therefore categorical:
        nb.Feature("Checking account status", distributions.Multinomial, data_point[0]),
            
        # Continuous and probably follows a power law distribution:
        nb.Feature("Duration in months", distributions.Exponential, float(data_point[1])),
        
        # Categorical:
        nb.Feature("Credit history", distributions.Multinomial, data_point[2]),
            
        # Categorical:
        nb.Feature("Purpose", distributions.Multinomial, data_point[3]),
            
        # Continuous and probably conforms to an approximate power law distribution:
        nb.Feature("Credit amount", distributions.Gaussian, float(data_point[4])),
            
        # Bucketed and therefore categorical:
        nb.Feature("Savings account status", distributions.Multinomial, data_point[5]),
            
        # Bucketed and therefore categorical:
        nb.Feature("Unemployment duration", distributions.Multinomial, data_point[6]),
            
        # Continuous and probably conforms to an approximate power law distribution:
        nb.Feature("Installment rate", distributions.Gaussian, float(data_point[7])),
            
        # Categorical:
        nb.Feature("Personal status", distributions.Multinomial, data_point[8]),
            
        # Categorical:
        nb.Feature("Other debtors", distributions.Multinomial, data_point[9]),
            
        # Continuous and probably conforms to an approximate power law distribution:
        nb.Feature("Present residence", distributions.Exponential, float(data_point[10])),
            
        # Categorical:
        nb.Feature("Property status", distributions.Multinomial, data_point[11]),
            
        # Continuous and probably conforms to an approximate power law distribution:
        nb.Feature("Age", distributions.Gaussian, float(data_point[12])),
            
        # Categorical:
        nb.Feature("Other installment plans", distributions.Multinomial, data_point[13]),
            
        # Categorical:
        nb.Feature("Housing", distributions.Multinomial, data_point[14]),
            
        # Continuous and probably conforms to an approximate power law distribution:
        nb.Feature("Number of credit cards", distributions.Exponential, float(data_point[15])),
            
        # Categorical:
        nb.Feature("Job", distributions.Multinomial, data_point[16]),
            
        # Continuous and probably conforms to an approximate power law distribution:
        nb.Feature("Number of people liable", distributions.Exponential, float(data_point[17])),
            
        # Categorical:
        nb.Feature("Telephone", distributions.Multinomial, data_point[18]),
            
        # Categorical:
        nb.Feature("Foreign worker", distributions.Multinomial, data_point[19])
    ]

def main():
    # Creating, training and testing the classifier:
    training_set, test_set = load_data()
    classifier = nb.NaiveBayesClassifier(featurizer)
    classifier.train(training_set[0], training_set[1])
    print "Accuracy = %s" % classifier.accuracy(test_set[0], test_set[1])

if __name__ == "__main__":
    main()
