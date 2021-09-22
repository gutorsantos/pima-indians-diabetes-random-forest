import pandas as pd
import numpy as np
from plotter import write_chart 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

def run_random_forest(x_train, x_test, y_train, y_test, est):
    classifier = RandomForestClassifier(n_estimators=est)   # creates the classifier to the Random Forest
    classifier.fit(x_train, y_train)                        # it realizes the training

    print('Results for ' + str(est) + ' estimators: ')

    # Check accuracy using score method
    accuracy = classifier.score(x_test, y_test)
    print('\tAccuracy :', accuracy)

    # Test accuracy using matrics
    y_pred = classifier.predict(x_test)                                     # it realizes the test and
    print('\t Accuracy Score:', metrics.accuracy_score(y_test,y_pred))      # compares with original data

    # Check confusion metrics
    print('\t Confusion Metrics :', metrics.confusion_matrix(y_test, y_pred))
    print('')

    write_chart(classifier, est)



def main():
    # Reads the data from file
    df = pd.read_csv('diabetes.csv')

    # The column that gives the class information if person is diabetic or not 
    # is the 'Outcome' column, that is the most important column to do our analysis 
    x = np.array(df.drop('Outcome', axis=1))
    y = np.array(df['Outcome'])

    # Split the dataframe between data to use in training and data to use in testing
    x_train_set, x_test_set, y_train_set, y_test_set = train_test_split(x, y, test_size = 0.2)

    run_random_forest(x_train_set, x_test_set, y_train_set, y_test_set, 10)
    run_random_forest(x_train_set, x_test_set, y_train_set, y_test_set, 100)
    # run_random_forest(x_train_set, x_test_set, y_train_set, y_test_set, 1000)

if __name__ == "__main__":
    main()