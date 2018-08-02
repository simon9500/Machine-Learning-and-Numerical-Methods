## Import and Data loading

import pickle
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt
import json
# Add SKLEARN modules
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale

### Function to plot the confusion matrix as a heatmap ###
# From http://scikit-learn.org/stable/auto_examples/model_selection/
# plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

### Display weather data ###
## Parameters:
# weather object
def display_weather_data(weather):
    print '#'*50
    print '# Inspect the Data'
    print '#'*50
    print '\n'

    # print data
    print 'Weather Data:'
    print weather.data

    # print number of entries
    print 'Number of entries: %s' % (weather.getNrEntries())

    # print target names
    print 'Number of targets: %s' % (weather.getNrTargets())

    print 'Target names: %s' % (weather.getTargetNames())

    # print features
    print 'Number of features: %s' % (weather.getNrFeatures())

    print 'Feature names: %s' % (weather.getFeatures())

### Grid search over the hyperparameters of the estimator for each scoring method ###
## Parameters:
# feature data, target values,
# list of scoring methods, tuned parameters for classifier,
# num of k-folds, the classifier, testing data size
## Returns:
# best parameters, classification reports,
# confusion matrices (for each scoring method)
# total running time, running times for each 'best model'
def grid_search(data, targets, target_names,
                scores, tuned_parameters, cv, classifier, test_size):
    start_time = time.time() # Set start time for grid search
    x = data
    y = targets
    # Make training and testing data sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size,
                                                        random_state=0)
    # Array to hold best hyperparameters for each scoring method
    best_params = []
    # Array to hold classification report for each scoring method
    classification_reports = []
    # Array to hold confusion matrices for each scoring method
    cnf_matrices = []
    # Array to hold the running times for the model
    # with the best parameters for each scoring method
    running_times = []

    # Evaluate the best parameters using a grid search for each scoring method
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print ""

        clf = GridSearchCV(classifier, tuned_parameters, cv=cv,
                           scoring='%s_macro' % score)
        clf.fit(x_train, y_train)

        print("Best parameters set found on development set:")
        print ""
        print(clf.best_params_)
        best_params.append(clf.best_params_)
        print ""
        print("Detailed classification report:")
        print ""
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print ""

        best_model_start_time = time.time()
        y_true, y_pred = y_test, clf.predict(x_test)
        best_model_end_time = time.time()

        print(classification_report(y_true, y_pred, target_names=target_names))
        print ""

        # Evaluation metrics
        classification_reports.append(classification_report(y_true, y_pred,
                                                            target_names=target_names))
        cnf_matrices.append(confusion_matrix(y_true, y_pred))
        # Running time for best model
        running_time_best = best_model_end_time - best_model_start_time
        running_times.append(running_time_best)

    end_time = time.time()
    running_time_gs = end_time - start_time # Running time for grid search
    print "Grid search running time = %fs" %(running_time_gs)
    return best_params, classification_reports, \
           cnf_matrices, running_time_gs, running_times

### Write evaluation data to file ###
## Parameters:
# weather object, best parameters, classification reports, total running time,
# running times for each 'best model', output file handle for writing, scoring methods
def write_evaluation_to_file(weather, best_params, classification_reports,
                             running_time_gs, running_times, evalfile, scores):
    evalfile.write('Features used\n\n')
    for w in weather.getFeatures():
        evalfile.write(w)
        evalfile.write('   ')
    evalfile.write('\n\nRunning time for grid search\n')
    evalfile.write(str(running_time_gs))
    for i in range(len(scores)):
        evalfile.write('\n\nScoring method\n')
        evalfile.write(scores[i])
        evalfile.write('\n\nBest Parameters\n')
        evalfile.write(json.dumps(best_params[i]))
        evalfile.write('\n\nClassification report\n')
        evalfile.write(classification_reports[i])
        evalfile.write('\n\nRunning time\n')
        evalfile.write(str(running_times[i]))

### Save confusion matrix image files for each scoring method ###
## Parameters:
# list of confusion matrices, scoring methods, weather object
def produce_confusion_matrices(cnf_matrices, scores, target_names):
    for i in range(len(scores)):
        plt.figure()
        plot_confusion_matrix(cnf_matrices[i], classes=target_names,
                              title='Confusion matrix, without normalization')
        plt.savefig('MLconf_matrix_%s.png' %scores[i])

## Main program
def main():
    # Load the weather data created by FeatureExtraction.py
    weather = pickle.load(open('data/mldata.p'))

    # Define the ML parameters
    scores = ['f1'] # Scoring methods to evaluate (of form 'METHOD_macro')
    cv = 5 # Number of k-folds for cross validation
    test_size = 0.2 # % size of testing data

    # Choose classifier to grid search
    classifier = RandomForestClassifier()
    # Parameter grid for the classifier
    tuned_parameters_default = {}

    # Scale the data for each individual feature to have 0 variance and 0 mean
    data = scale(weather.data)
    # Extract weather type as a float number for each data point in weather
    targets = weather.target.astype(np.float)
    # List of weather types
    target_names = weather.getTargetNames()

    # Perform the grid searches
    best_params, classification_reports, cnf_matrices, running_time_gs, running_times = \
        grid_search(data, targets, target_names, scores,
                    tuned_parameters_default, cv, classifier, test_size)

    # Write evaluation information to file
    evalfname = "MLeval.txt"
    evalfile = open(evalfname, 'w')
    write_evaluation_to_file(weather, best_params, classification_reports,
                             running_time_gs, running_times, evalfile, scores)
    evalfile.close()

    # Save confusion matrices
    produce_confusion_matrices(cnf_matrices, scores, target_names)

main() # Run program