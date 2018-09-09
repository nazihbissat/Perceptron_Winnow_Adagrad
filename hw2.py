import sklearn
import numpy as np
import os
import matplotlib.pyplot as plt
from  sklearn import svm
from sklearn import feature_extraction


class Classifier(object):
    def __init__(self, algorithm, x_train, y_train, iterations=1, averaged=False, eta=1.5, alpha=1.1):

        # Get features from examples; this line figures out what features are present in
        # the training data, such as 'w-1=dog' or 'w+1=cat'
        features = {feature for xi in x_train for feature in xi.keys()}

        dvsparse = sklearn.feature_extraction.DictVectorizer(sparse=True)
        dvdense = sklearn.feature_extraction.DictVectorizer(sparse=False)

        if (algorithm == 'Perceptron' and averaged == False):
            # Initialize w, bias
            self.w, self.w['bias'] = {feature: 0.0 for feature in features}, 0.0
            # Iterate over the training data n times
            for i in range(iterations):
                # Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    # Update weights if there is a misclassification
                    if yi != y_hat:
                        for feature, value in xi.items():
                            self.w[feature] = self.w[feature] + yi * value
                        self.w['bias'] = self.w['bias'] + yi

        elif (algorithm == 'Winnow' and averaged == False):
            # Initialize w, bias
            self.w, self.w['bias'] = {feature: 1.0 for feature in features}, -len(features)
            # Iterate over the training data n times
            for i in range(iterations):
                # Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    # Update weights if there is a misclassification
                    if yi != y_hat:
                        for feature, value in xi.items():
                            self.w[feature] = self.w[feature] * alpha ** (yi * value)

        elif (algorithm == 'Adagrad' and averaged == False):
            # Initialize w, bias
            self.w, self.w['bias'] = {feature: 0.0 for feature in features}, 0.0
            # Initialize a dictionary of zeros in order to keep track of the running sum of squares of gradient over
            # iterations
            self.grad_sq, self.grad_sq['bias'] = {feature: 0.0 for feature in features}, 0.0
            # Iterate over the training data n times
            for i in range(iterations):
                # Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    s =(sum([self.w[feature] * value for feature, value in xi.items()]) + self.w['bias'])
                    hinge = yi * s
                    for feature, value in xi.items():
                        self.grad_sq[feature] = self.grad_sq[feature] + ((yi * value) ** 2)
                    self.grad_sq['bias'] = self.grad_sq['bias'] + ((yi) ** 2)
                    # Update weights if the hinge loss condition is satisfied
                    if hinge <= 1:
                        for feature, value in xi.items():
                            self.w[feature] = self.w[feature] + eta * yi * value / np.sqrt(self.grad_sq[feature])
                        self.w['bias'] = self.w['bias'] + (eta * yi / np.sqrt(self.grad_sq['bias']))

        elif (algorithm == 'Perceptron Inefficient' and averaged == True):
            # Initialize w, bias
            self.w, self.w['bias'] = {feature: 0.0 for feature in features}, 0.0
            # Initialize consistency count, number of mistakes, and initial weight vector
            num_mistakes = 0
            c = 0
            self.w_cum, self.w_cum['bias'] = {feature: 0.0 for feature in features}, 0.0
            mistake = False
            # Iterate over the training data n times
            for i in range(iterations):
                # Check each training example
                for i in range(len(y_train)):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    # Update weights if there is a misclassification
                    if yi != y_hat:
                        for feature in features:
                            self.w_cum[feature] = self.w_cum[feature] + c * (self.w[feature])
                        self.w_cum['bias'] = self.w_cum['bias'] + c * (self.w['bias'])
                        for feature, value in xi.items():
                            self.w[feature] = self.w[feature] + yi * value
                        self.w['bias'] = self.w['bias'] + yi
                        c = 1
                        num_mistakes += 1
                        mistake = True
                    else:
                        c += 1
                        mistake = False

            if mistake:
                for feature in features:
                    self.w[feature] = self.w_cum[feature]
                self.w['bias'] = self.w_cum['bias']
            else:
                for feature in features:
                    self.w[feature] = c * self.w[feature] + self.w_cum[feature]
                self.w['bias'] = c * self.w['bias'] + self.w_cum['bias']

        elif (algorithm == 'Perceptron Efficient' and averaged == True):
            # Initialize w, bias
            self.w, self.w['bias'] = {feature: 0.0 for feature in features}, 0.0
            # Initialize consistency count, number of mistakes, and initial weight vector
            c = 1
            self.delta_cum, self.delta_cum['bias'] = {feature: 0.0 for feature in features}, 0.0
            # Iterate over the training data n times
            for i in range(iterations):
                # Check each training example
                for i in range(len(y_train)):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    # Update weights if there is a misclassification
                    if yi != y_hat:
                        for feature, value in xi.items():
                            self.w[feature] = self.w[feature] + yi * eta * value
                            self.delta_cum[feature] = self.delta_cum[feature] + c * (yi * eta * value)
                        self.w['bias'] = self.w['bias'] + yi * eta
                        self.delta_cum['bias'] = self.delta_cum['bias'] + c * (yi * eta)
                    c += 1

            for feature in features:
                self.w[feature] = self.w[feature] - (1.0 / c) * self.delta_cum[feature]
            self.w['bias'] = self.w['bias'] - (1.0 / c) * self.delta_cum['bias']


        elif (algorithm == 'Winnow' and averaged == True):
            # Initialize w, bias
            self.w, self.w['bias'] = {feature: 1.0 for feature in features}, (-len(features))
            # Initialize consistency count and initial weight vector
            c = 0
            self.w_cum, self.w_cum['bias'] = {feature: 0.0 for feature in features}, 0.0
            mistake = False
            # Iterate over the training data n times
            for i in range(iterations):
                # Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    # Update weights if there is a misclassification
                    if yi != y_hat:
                        for feature in features:
                            self.w_cum[feature] = self.w_cum[feature] + c * (self.w[feature])
                        self.w_cum['bias'] = self.w_cum['bias'] + c * (self.w['bias'])
                        for feature, value in xi.items():
                            self.w[feature] = self.w[feature] * (alpha ** (yi * value))
                        c = 1
                        mistake = True
                    else:
                        c += 1
                        mistake = False

            if mistake:
                for feature in features:
                    self.w[feature] = self.w_cum[feature]
                self.w['bias'] = self.w_cum['bias']
            else:
                for feature in features:
                    self.w[feature] = c * self.w[feature] + self.w_cum[feature]
                self.w['bias'] = self.w_cum['bias'] + c * (self.w['bias'])

        elif (algorithm == 'Adagrad' and averaged == True):
            # Initialize w, bias
            self.w, self.w['bias'] = {feature: 0.0 for feature in features}, 0
            # Initialize a vector of zeros in order to keep track of the running sum of squares of the gradient over
            # iterations
            self.grad_sq, self.grad_sq['bias'] = {feature: 0.0 for feature in features}, 0.0
            # Initialize consistency count, number of mistakes, and initial weight vector
            c = 0
            self.w_cum, self.w_cum['bias'] = {feature: 0.0 for feature in features}, 0.0
            mistake = False
            # Iterate over the training data n times
            for i in range(iterations):
                # Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    s = (sum([self.w[feature] * value for feature, value in xi.items()]) + self.w['bias'])
                    hinge = yi * s
                    for feature, value in xi.items():
                        self.grad_sq[feature] = self.grad_sq[feature] + ((yi * value) ** 2)
                    self.grad_sq['bias'] = self.grad_sq['bias'] + ((yi) ** 2)
                    # Update weights if the hinge loss condition is satisfied
                    if hinge < 1:
                        for feature in features:
                            self.w_cum[feature] = self.w_cum[feature] + c * (self.w[feature])
                        self.w_cum['bias'] = self.w_cum['bias'] + c * (self.w['bias'])
                        for feature, value in xi.items():
                            self.w[feature] = self.w[feature] + eta * yi * value / np.sqrt(self.grad_sq[feature])
                        self.w['bias'] = self.w['bias'] + (eta * yi / np.sqrt(self.grad_sq['bias']))
                        c = 1
                        mistake = True
                    else:
                        c += 1
                        mistake = False

            if mistake:
                for feature in features:
                    self.w[feature] = self.w_cum[feature]
                self.w['bias'] = self.w_cum['bias']
            else:
                for feature in features:
                    self.w[feature] = c * self.w[feature] + self.w_cum[feature]
                self.w['bias'] = self.w_cum['bias'] + c * (self.w['bias'])


    def predict(self, x):
        s = sum([self.w[feature] * value for feature, value in x.items()]) + self.w['bias']
        return 1 if s > 0 else -1

    def predict_np(self, x):
        dvdense = sklearn.feature_extraction.DictVectorizer(sparse=False)
        x.append(1)
        s = sum(np.multiply(dvdense.fit_transform(self.w), x))
        return 1 if s > 0 else -1


# Parse the real-world data to generate features,
# Returns a list of tuple lists
def parse_real_data(path):
    # List of tuples for each sentence
    data = []
    for filename in os.listdir(path):
        with open(path + filename, 'r') as file:
            sentence = []
            for line in file:
                if line == '\n':
                    data.append(sentence)
                    sentence = []
                else:
                    sentence.append(tuple(line.split()))
    return data


# Returns a list of labels
def parse_synthetic_labels(path):
    # List of tuples for each sentence
    labels = []
    with open(path + 'y.txt', 'rb') as file:
        for line in file:
            labels.append(int(line.strip()))
    return labels


# Returns a list of features
def parse_synthetic_data(path):
    # List of tuples for each sentence
    data = []
    with open(path + 'x.txt') as file:
        features = []
        for line in file:
            # print('Line:', line)
            for ch in line:
                if ch == '[' or ch.isspace():
                    continue
                elif ch == ']':
                    data.append(features)
                    features = []
                else:
                    features.append(int(ch))
    return data


if __name__ == '__main__':
    print('Loading data...')
    # Load data from folders.
    # Real world data - lists of tuple lists
    news_train_data = parse_real_data('Data/Real-World/CoNLL/train/')
    news_dev_data = parse_real_data('Data/Real-World/CoNLL/dev/')
    news_test_data = parse_real_data('Data/Real-World/CoNLL/test/')
    email_dev_data = parse_real_data('Data/Real-World/Enron/dev/')
    email_test_data = parse_real_data('Data/Real-World/Enron/test/')

    # #Load dense synthetic data
    syn_dense_train_data = parse_synthetic_data('Data/Synthetic/Dense/train/')
    syn_dense_train_labels = parse_synthetic_labels('Data/Synthetic/Dense/train/')
    syn_dense_dev_data = parse_synthetic_data('Data/Synthetic/Dense/dev/')
    syn_dense_dev_labels = parse_synthetic_labels('Data/Synthetic/Dense/dev/')
    syn_dense_test_data = parse_synthetic_data('Data/Synthetic/Dense/test/')

    # Load sparse synthetic data
    syn_sparse_train_data = parse_synthetic_data('Data/Synthetic/Sparse/train/')
    syn_sparse_train_labels = parse_synthetic_labels('Data/Synthetic/Sparse/train/')
    syn_sparse_dev_data = parse_synthetic_data('Data/Synthetic/Sparse/dev/')
    syn_sparse_dev_labels = parse_synthetic_labels('Data/Synthetic/Sparse/dev/')
    syn_sparse_test_data = parse_synthetic_data('Data/Synthetic/Sparse/test/')

    # Convert to sparse dictionary representations.
    # Examples are a list of tuples, where each tuple consists of a dictionary
    # and a lable. Each dictionary contains a list of features and their values,
    # i.e a feature is included in the dictionary only if it provides information.

    # You can use sklearn.feature_extraction.DictVectorizer() to convert these into
    # scipy.sparse format to train SVM, or for your Perceptron implementation.
    print('Converting Synthetic data...')
    syn_dense_train = zip(*[({'x' + str(i): syn_dense_train_data[j][i]
                              for i in range(len(syn_dense_train_data[j])) if syn_dense_train_data[j][i] == 1},
                             syn_dense_train_labels[j])
                            for j in range(len(syn_dense_train_data))])
    syn_dense_train_x, syn_dense_train_y = syn_dense_train
    syn_dense_dev = zip(*[({'x' + str(i): syn_dense_dev_data[j][i]
                            for i in range(len(syn_dense_dev_data[j])) if syn_dense_dev_data[j][i] == 1},
                           syn_dense_dev_labels[j])
                          for j in range(len(syn_dense_dev_data))])
    syn_dense_dev_x, syn_dense_dev_y = syn_dense_dev
    syn_dense_test_x = *({'x' + str(i): syn_dense_test_data[j][i]
                            for i in range(len(syn_dense_test_data[j])) if syn_dense_test_data[j][i] == 1}
                            for j in range(len(syn_dense_test_data))),



    syn_sparse_train = zip(*[({'x' + str(i): syn_sparse_train_data[j][i]
                               for i in range(len(syn_sparse_train_data[j])) if syn_sparse_train_data[j][i] == 1},
                              syn_sparse_train_labels[j])
                             for j in range(len(syn_sparse_train_data))])
    syn_sparse_train_x, syn_sparse_train_y = syn_sparse_train
    syn_sparse_dev = zip(*[({'x' + str(i): syn_sparse_dev_data[j][i]
                             for i in range(len(syn_sparse_dev_data[j])) if syn_sparse_dev_data[j][i] == 1},
                            syn_sparse_dev_labels[j])
                           for j in range(len(syn_sparse_dev_data))])
    syn_sparse_dev_x, syn_sparse_dev_y = syn_sparse_dev

    syn_sparse_test_x = *({'x' + str(i): syn_sparse_test_data[j][i]
                            for i in range(len(syn_sparse_test_data[j])) if syn_sparse_test_data[j][i] == 1}
                             for j in range(len(syn_sparse_test_data))),


    # Feature extraction
    print('Extracting features from real-world data...')
    news_train_y = []
    news_train_x = []
    train_features = set([])
    for sentence in news_train_data:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(2, len(padded) - 2):
            news_train_y.append(1 if padded[i][1] == 'I' else -1)
            feat1 = 'w-1=' + str(padded[i - 1][0])
            feat2 = 'w+1=' + str(padded[i + 1][0])
            feat3 = 'w-2=' + str(padded[i - 2][0])
            feat4 = 'w+2=' + str(padded[i + 2][0])
            feat5 = 'w-2&w-1=' + str(padded[i - 2][0]) + str(padded[i - 1][0])
            feat6 = 'w1&w2=' + str(padded[i + 1][0]) + str(padded[i + 2][0])
            feat7 = 'w-1&w1=' + str(padded[i - 1][0]) + str(padded[i + 1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7]
            train_features.update(feats)
            feats = {feature: 1 for feature in feats}
            news_train_x.append(feats)
    news_dev_y = []
    news_dev_x = []
    for sentence in news_dev_data:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(2, len(padded) - 2):
            news_dev_y.append(1 if padded[i][1] == 'I' else -1)
            feat1 = 'w-1=' + str(padded[i - 1][0])
            feat2 = 'w+1=' + str(padded[i + 1][0])
            feat3 = 'w-2=' + str(padded[i - 2][0])
            feat4 = 'w+2=' + str(padded[i + 2][0])
            feat5 = 'w-2&w-1=' + str(padded[i - 2][0]) + str(padded[i - 1][0])
            feat6 = 'w1&w2=' + str(padded[i + 1][0]) + str(padded[i + 2][0])
            feat7 = 'w-1&w1=' + str(padded[i - 1][0]) + str(padded[i + 1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7]
            feats = {feature: 1 for feature in feats if feature in train_features}
            news_dev_x.append(feats)
    news_test_x = []
    for sentence in news_test_data:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(2, len(padded) - 2):
            feat1 = 'w-1=' + str(padded[i - 1][0])
            feat2 = 'w+1=' + str(padded[i + 1][0])
            feat3 = 'w-2=' + str(padded[i - 2][0])
            feat4 = 'w+2=' + str(padded[i + 2][0])
            feat5 = 'w-2&w-1=' + str(padded[i - 2][0]) + str(padded[i - 1][0])
            feat6 = 'w1&w2=' + str(padded[i + 1][0]) + str(padded[i + 2][0])
            feat7 = 'w-1&w1=' + str(padded[i - 1][0]) + str(padded[i + 1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7]
            feats = {feature: 1 for feature in feats if feature in train_features}
            news_test_x.append(feats)

    email_dev_y = []
    email_dev_x = []
    for sentence in email_dev_data:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(2, len(padded) - 2):
            email_dev_y.append(1 if padded[i][1] == 'I' else -1)
            feat1 = 'w-1=' + str(padded[i - 1][0])
            feat2 = 'w+1=' + str(padded[i + 1][0])
            feat3 = 'w-2=' + str(padded[i - 2][0])
            feat4 = 'w+2=' + str(padded[i + 2][0])
            feat5 = 'w-2&w-1=' + str(padded[i - 2][0]) + str(padded[i - 1][0])
            feat6 = 'w1&w2=' + str(padded[i + 1][0]) + str(padded[i + 2][0])
            feat7 = 'w-1&w1=' + str(padded[i - 1][0]) + str(padded[i + 1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7]
            feats = {feature: 1 for feature in feats if feature in train_features}
            email_dev_x.append(feats)
    email_test_x = []
    for sentence in email_test_data:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(2, len(padded) - 2):
            feat1 = 'w-1=' + str(padded[i - 1][0])
            feat2 = 'w+1=' + str(padded[i + 1][0])
            feat3 = 'w-2=' + str(padded[i - 2][0])
            feat4 = 'w+2=' + str(padded[i + 2][0])
            feat5 = 'w-2&w-1=' + str(padded[i - 2][0]) + str(padded[i - 1][0])
            feat6 = 'w1&w2=' + str(padded[i + 1][0]) + str(padded[i + 2][0])
            feat7 = 'w-1&w1=' + str(padded[i - 1][0]) + str(padded[i + 1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7]
            feats = {feature: 1 for feature in feats if feature in train_features}
            email_test_x.append(feats)


# Function to tune promotion/demotion parameter alpha for winnow using grid search, given a training set and a
# data type, returns the optimal promotion/demotion rate
def hyperparam_winnow(x_train, y_train, type, averaged):
    alphas = [1.1, 1.01, 1.005, 1.0005, 1.0001]

    if type == 'syn_dense':
        dev_set_x = syn_dense_dev_x
        dev_set_y = syn_dense_dev_y

    else:
        dev_set_x = syn_sparse_dev_x
        dev_set_y = syn_sparse_dev_y

    dev_accuracies_winnow = []
    train_accuracies_winnow = []

    for alpha in alphas:
        clf = Classifier('Winnow', x_train, y_train, averaged=averaged, alpha=alpha, iterations=20)
        dev_accuracy = sum(
            [1 for i in range(len(dev_set_y)) if clf.predict(dev_set_x[i]) == dev_set_y[i]]) / len(
            dev_set_y) * 100
        train_accuracy = sum([1 for i in range(len(y_train)) if clf.predict(x_train[i]) == y_train[i]]) / len(
            y_train) * 100
        dev_accuracies_winnow.append(dev_accuracy)
        train_accuracies_winnow.append(train_accuracy)

    optimal_alpha = alphas[dev_accuracies_winnow.index(max(dev_accuracies_winnow))]
    return optimal_alpha


# Function to tune promotion/demotion parameter alpha for winnow using grid search, given a training set and a
# data type, returns the optimal promotion/demotion rate
def hyperparam_adagrad(x_train, y_train, type, averaged):
    etas = [1.5, 0.25, 0.03, 0.005, 0.001]

    if type == 'syn_dense':
        dev_set_x = syn_dense_dev_x
        dev_set_y = syn_dense_dev_y

    else:
        dev_set_x = syn_sparse_dev_x
        dev_set_y = syn_sparse_dev_y

    dev_accuracies_adagrad = []
    train_accuracies_adagrad = []

    for eta in etas:
        clf = Classifier('Adagrad', x_train, y_train, averaged=averaged, eta=eta, iterations=20)
        dev_accuracy = sum(
            [1 for i in range(len(dev_set_y)) if clf.predict(dev_set_x[i]) == dev_set_y[i]]) / len(
            dev_set_y) * 100
        train_accuracy = sum([1 for i in range(len(y_train)) if clf.predict(x_train[i]) == y_train[i]]) / len(
            y_train) * 100
        dev_accuracies_adagrad.append(dev_accuracy)
        train_accuracies_adagrad.append(train_accuracy)

    optimal_eta = etas[dev_accuracies_adagrad.index(max(dev_accuracies_adagrad))]
    return optimal_eta

# Function to convert a given dataset to a sparse dictionary
def data_to_dict(type, size):
    if type == 'syn_dense':
        train_data = syn_dense_train_data[0:size][:]
        train_labels = syn_dense_train_labels[0:size][:]
    else:
        train_data = syn_sparse_train_data[0:size][:]
        train_labels = syn_sparse_train_labels[0:size][:]
    train = zip(*[({'x' + str(i): train_data[j][i]
                              for i in range(len(train_data[j])) if train_data[j][i] == 1},
                            train_labels[j])
                            for j in range(len(train_data))])
    train_x, train_y = train
    return [train_x, train_y]

# Function to split training data into 10 subsamples of sizes 500 to 5000 in increments of 500 and add these 10 new
# datasets in addition to the full training dataset to a list of tuples
def learning_curve_data(type):
    datasets = []
    range = np.arange(500, 5001, 500)
    for r in range:
        datasets.append(data_to_dict(type, r))
    if type == 'syn_dense':
        datasets.append([syn_dense_train_x, syn_dense_train_y])
    else:
        datasets.append([syn_sparse_train_x, syn_sparse_train_y])
    return datasets

# Function to plot learning curves given training data and an algorithm
def plot_learning_curve(datasets, type):

    if type == 'syn_dense':
        dev_x = syn_dense_dev_x
        dev_y = syn_dense_dev_y
        dict_to_vec = sklearn.feature_extraction.DictVectorizer(sparse=False)
    else:
        dev_x = syn_sparse_dev_x
        dev_y = syn_sparse_dev_y
        dict_to_vec = sklearn.feature_extraction.DictVectorizer(sparse=True)

    svm_dev_y = np.asarray(dev_y)

    i = 0
    xi = np.arange(0, 11, 1)

    models = [('Perceptron', False), ('Winnow', False), ('Adagrad', False), ('Perceptron Efficient', True),
              ('Winnow', True), ('Adagrad', True), ('SVM', False)]

    for model in models:
        plot_xs = []
        plot_ys = []

        for i in xi:
            train_x = datasets[i][0]
            train_y = datasets[i][1]

            plot_xs.append(len(datasets[i][1]))

            if (model[0] == 'SVM'):
                svml = sklearn.svm.LinearSVC()
                x_train_vec = dict_to_vec.fit_transform(train_x)
                svm_train_y = np.asarray(train_y)
                svml.fit(x_train_vec, svm_train_y)
                dense_x_dev_vec = dict_to_vec.fit_transform(dev_x)
                # accuracy = sum([1 for i in range(len(train_y)) if
                #             p.predict(dense_x_dev_vec[i]) == dev_y[i]]) / len(dev_y) * 100
                accuracy = svml.score(dense_x_dev_vec, svm_dev_y) * 100
                plot_ys.append(accuracy)
            else:
                p = Classifier(model[0], train_x, train_y, averaged=model[1], iterations=10)
                accuracy = sum(
                    [1 for i in range(len(dev_y)) if p.predict(dev_x[i]) == dev_y[i]]) / len(
                    dev_y) * 100
                plot_ys.append(accuracy)

        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy on Development Set (%)")
        plt.xticks(xi, plot_xs)
        if model[1]:
            plt.plot(xi, plot_ys, marker=".", linestyle="-", label="Averaged " + model[0])
        else:
            plt.plot(xi, plot_ys, marker=".", linestyle="-", label=model[0])

    plt.legend()
    plt.savefig(type + ".png")

# Print results
print('\nTesting Algorithm Accuracies')

datasets_dense = learning_curve_data('syn_dense')
datasets_sparse = learning_curve_data('syn_sparse')

# Hyperparameter Search for Winnow and Adagrad on both Synthetic Dense and Synthetic Sparse
# opt_eta_dense = hyperparam_adagrad(syn_dense_train_x, syn_dense_train_y, 'syn_dense', False)
# print('The optimal eta for the Dense Synthetic Data is: ', opt_eta_dense)

# opt_alpha_dense = hyperparam_winnow(syn_dense_train_x, syn_dense_train_y, 'syn_dense', False)
# print('The optimal alpha for the Dense Synthetic Data is: ', opt_alpha_dense)

# opt_alpha_sparse = hyperparam_winnow(syn_sparse_train_x, syn_sparse_train_y, 'syn_sparse', False)
# print('The optimal alpha for the Sparse Synthetic Data is: ', opt_alpha_sparse)

# opt_eta_sparse = hyperparam_adagrad(syn_sparse_train_x, syn_sparse_train_y, 'syn_sparse', False)
# print('The optimal eta for the Sparse Synthetic Data is: ', opt_eta_sparse)

# # Test Perceptron on Dense Synthetic
# p = Classifier('Perceptron', syn_dense_train_x, syn_dense_train_y)
# accuracy = sum(
#     [1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]]) / len(
#     syn_dense_dev_y) * 100
# print('Perceptron Syn Dense Dev Accuracy:', accuracy)

# # Test Inefficient Averaged Perceptron on Dense Synthetic
# p = Classifier('Perceptron Inefficient', syn_dense_train_x, syn_dense_train_y, averaged=True)
# accuracy = sum(
#     [1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]]) / len(
#     syn_dense_dev_y) * 100
# print('Inefficient Averaged Perceptron Syn Dense Dev Accuracy:', accuracy)
#
# # Test Efficient Averaged Perceptron on Dense Synthetic
# p = Classifier('Perceptron Efficient', syn_dense_train_x, syn_dense_train_y, averaged=True)
# accuracy = sum(
#     [1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]]) / len(
#     syn_dense_dev_y) * 100
# print('Efficient Averaged Perceptron Syn Dense Dev Accuracy:', accuracy)
#
# # Predict synthetic dense test labels using efficient averaged perceptron
# openfile = open('p-dense.txt', 'w')
# for i in range(len(syn_dense_test_data)):
#     openfile.write(str(p.predict(syn_dense_test_x[i])) + '\n')
# openfile.close()
#
# # Test Winnow on Dense Synthetic
# p = Classifier('Winnow', syn_dense_train_x, syn_dense_train_y)
# accuracy = sum(
#     [1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]]) / len(
#     syn_dense_dev_y) * 100
# print('Winnow Syn Dense Dev Accuracy:', accuracy)
#
# # Test Averaged Winnow on Dense Synthetic
# p = Classifier('Winnow', syn_dense_train_x, syn_dense_train_y, averaged=True)
# accuracy = sum(
#     [1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]]) / len(
#     syn_dense_dev_y) * 100
# print('Averaged Winnow Syn Dense Dev Accuracy:', accuracy)
#
# # Test Adagrad on Dense Synthetic
# p = Classifier('Adagrad', syn_dense_train_x, syn_dense_train_y)
# accuracy = sum(
#     [1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]]) / len(
#     syn_dense_dev_y) * 100
# print('Adagrad Syn Dense Dev Accuracy:', accuracy)
#
# # Test Averaged Adagrad on Dense Synthetic
# p = Classifier('Adagrad', syn_dense_train_x, syn_dense_train_y, averaged=True)
# accuracy = sum(
#     [1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]]) / len(
#     syn_dense_dev_y) * 100
# print('Averaged Adagrad Syn Dense Dev Accuracy:', accuracy)
#
# # Test SVM on Dense Synthetic
# svm_syn_dense = sklearn.svm.LinearSVC()
# dict_to_vec = sklearn.feature_extraction.DictVectorizer(sparse=False)
# dense_x_train_vec = dict_to_vec.fit_transform(syn_dense_train_x)
# dense_x_dev_vec = dict_to_vec.transform(syn_dense_dev_x)
# svm_syn_dense_train_y = np.asarray(syn_dense_train_y)
# svm_syn_dense_dev_y = np.asarray(syn_dense_dev_y)
# svm_syn_dense.fit(dense_x_train_vec, syn_dense_train_y)
# accuracy = svm_syn_dense.score(dense_x_dev_vec, svm_syn_dense_dev_y) * 100
# print('SVM Syn Dense Dev Accuracy:', accuracy)
#
# # Predict synthetic dense test labels using SVM
# dense_x_test_vec = dict_to_vec.transform(syn_dense_test_x)
# openfile = open('svm-dense.txt', 'w')
# for i in range((dense_x_test_vec.shape[0])):
#     openfile.write(str(svm_syn_dense.predict(dense_x_test_vec[i])[0]) + '\n')
# openfile.close()
#
# # Test Perceptron on Sparse Synthetic
# p = Classifier('Perceptron', syn_sparse_train_x, syn_sparse_train_y)
# accuracy = sum(
#     [1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]]) / len(
#     syn_sparse_dev_y) * 100
# print('Perceptron Syn Sparse Dev Accuracy:', accuracy)
#
# # Test Efficient Averaged Perceptron on Sparse Synthetic
# p = Classifier('Perceptron Efficient', syn_sparse_train_x, syn_sparse_train_y, averaged=True)
# accuracy = sum(
#     [1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]]) / len(
#     syn_sparse_dev_y) * 100
# print('Efficient Averaged Perceptron Syn Sparse Dev Accuracy:', accuracy)
# # Predict synthetic dense test labels using efficient averaged perceptron
# openfile = open('p-sparse.txt', 'w')
# for i in range(len(syn_sparse_test_data)):
#     openfile.write(str(p.predict(syn_sparse_test_x[i])) + '\n')
# openfile.close()
#
# # Test Inefficient Averaged Perceptron on Sparse Synthetic
# p = Classifier('Perceptron Inefficient', syn_sparse_train_x, syn_sparse_train_y, averaged=True)
# accuracy = sum(
#     [1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]]) / len(
#     syn_sparse_dev_y) * 100
# print('Inefficient Averaged Perceptron Syn Sparse Dev Accuracy:', accuracy)
#
# # Test Winnow on Sparse Synthetic
# p = Classifier('Winnow', syn_sparse_train_x, syn_sparse_train_y)
# accuracy = sum(
#     [1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]]) / len(
#     syn_sparse_dev_y) * 100
# print('Winnow Syn Sparse Dev Accuracy:', accuracy)
#
# # Test Averaged Winnow on Sparse Synthetic
# p = Classifier('Winnow', syn_sparse_train_x, syn_sparse_train_y, averaged=True)
# accuracy = sum(
#     [1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]]) / len(
#     syn_sparse_dev_y) * 100
# print('Averaged Winnow Syn Dense Dev Accuracy:', accuracy)
#
# Test Adagrad on Sparse Synthetic
# p = Classifier('Adagrad', syn_sparse_train_x, syn_sparse_train_y)
# accuracy = sum(
#     [1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]]) / len(
#     syn_sparse_dev_y) * 100
# print('Adagrad Syn Sparse Dev Accuracy:', accuracy)
#
# # Test Averaged Adagrad on Sparse Synthetic
# p = Classifier('Adagrad', syn_sparse_train_x, syn_sparse_train_y, averaged=True)
# accuracy = sum(
#     [1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]]) / len(
#     syn_sparse_dev_y) * 100
# print('Averaged Adagrad Syn Sparse Dev Accuracy:', accuracy)
#
# # Test SVM on Sparse Synthetic
# svm_sparse = sklearn.svm.LinearSVC()
# dict_to_vec = sklearn.feature_extraction.DictVectorizer(sparse=True)
# sparse_x_train_vec = dict_to_vec.fit_transform(syn_sparse_train_x)
# sparse_x_dev_vec = dict_to_vec.transform(syn_sparse_dev_x)
# svm_syn_sparse_train_y = np.asarray(syn_sparse_train_y)
# svm_syn_sparse_dev_y = np.asarray(syn_sparse_dev_y)
# svm_sparse.fit(sparse_x_train_vec, syn_sparse_train_y)
# accuracy = svm_sparse.score(sparse_x_dev_vec, svm_syn_sparse_dev_y) * 100
#print('SVM Syn Sparse Dev Accuracy:', accuracy)
#
# # Predict synthetic sparse test labels using SVM
# sparse_x_test_vec = dict_to_vec.transform(syn_sparse_test_x)
# openfile = open('svm-sparse.txt', 'w')
# for i in range((sparse_x_test_vec.shape[0])):
#     openfile.write(str(svm_sparse.predict(sparse_x_test_vec[i])[0]) + '\n')
# openfile.close()
#
# # Plot Learning Curves for Dense and Sparse Synthetic Datasets
# plot_learning_curve(datasets_dense, 'syn_dense')
# plot_learning_curve(datasets_sparse, 'syn_sparse')
#
# # Test Averaged Perceptron on Real World Data (CoNLL)
# p = Classifier('Perceptron Efficient', news_train_x, news_train_y, iterations=20, averaged=True)
# accuracy = sum([1 for i in range(len(news_dev_y)) if p.predict(news_dev_x[i]) == news_dev_y[i]]) / len(
#     news_dev_y) * 100
# print('Averaged Perceptron News Dev Accuracy:', accuracy)
# # Predict CoNLL test labels using efficient averaged perceptron
# openfile = open('p-conll.txt', 'w')
# for i in range(len(news_test_x)):
#     label = p.predict(news_test_x[i])
#     if label == 1:
#         openfile.write('I\n')
#     else:
#         openfile.write('O\n')
# openfile.close()
# # Predict Enron test labels using efficient averaged perceptron
# openfile = open('p-enron.txt', 'w')
# for i in range(len(email_test_x)):
#     label = p.predict(email_test_x[i])
#     if label == 1:
#         openfile.write('I\n')
#     else:
#         openfile.write('O\n')
# openfile.close()
#
# # Test SVM on Real World Data (CoNLL)
# svm_conll = sklearn.svm.LinearSVC()
# dict_to_vec = sklearn.feature_extraction.DictVectorizer(sparse=True)
# news_x_train_vec = dict_to_vec.fit_transform(news_train_x)
# news_x_dev_vec = dict_to_vec.transform(news_dev_x)
# svm_news_train_y = np.asarray(news_train_y)
# svm_news_dev_y = np.asarray(news_dev_y)
# svm_conll.fit(news_x_train_vec, svm_news_train_y)
# accuracy1 = svm_conll.score(news_x_dev_vec, svm_news_dev_y) * 100
# print('SVM News Dev Accuracy:', accuracy1)
# # Predict CoNLL test labels using SVM
# news_x_test_vec = dict_to_vec.transform(news_test_x)
# openfile = open('svm-conll.txt', 'w')
# for i in range((news_x_test_vec.shape[0])):
#     label = svm_conll.predict(news_x_test_vec[i])[0]
#     if label == 1:
#         openfile.write('I\n')
#     else:
#         openfile.write('O\n')
# openfile.close()
#
# # Test Previously Trained Averaged Perceptron on Real World Data (Enron)
# email_x_dev_vec = dict_to_vec.transform(email_dev_x)
# svm_email_dev_y = np.asarray(email_dev_y)
# accuracy1 = svm_conll.score(email_x_dev_vec, svm_email_dev_y) * 100
# print('SVM Email Dev Accuracy:', accuracy1)
# # Predict CoNLL test labels using SVM
# email_x_test_vec = dict_to_vec.transform(email_test_x)
# openfile = open('svm-enron.txt', 'w')
# for i in range((email_x_test_vec.shape[0])):
#     label = svm_conll.predict(email_x_test_vec[i])[0]
#     if label == 1:
#         openfile.write('I\n')
#     else:
#         openfile.write('O\n')
# openfile.close()