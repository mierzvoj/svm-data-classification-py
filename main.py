import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

# Lukasz Cettler s20168
# Wojciech Mierzejewski s21617

# This is the example of the wine quality and water potability classification labeling them good or bad.
# Wines with quality above 6 are classified as good and wines with quality 6 and below are classified as bad.
# Water with potability 1 is classified as bad else is good.


def wine_quality(wine_data):
    # Converting data to a dictionary. All the columns in dataFrame become elements of dictionary.
    wine_data = dict(wine_data)
    n = len(wine_data['quality'])

    # The values held by each dictionary is a numpy array. We convert them to a list for replacement.
    wine_data['quality'] = wine_data['quality'].tolist()

    # Here we are replacing each data entry with either good or bad.
    for i in range(n):
        if wine_data['quality'][i] < 7:
            wine_data['quality'][i] = "bad"
        else:
            wine_data['quality'][i] = "good"

            # Thean again we are converting it to a dataframe.
    wine_data = pd.DataFrame(wine_data)
    return wine_data

def water_potability(water_data):
    # Converting data to a dictionary. All the columns in dataFrame become elements of dictionary.
    water_data = dict(water_data)
    n = len(water_data['Potability'])

    # The values held by each dictionary is a numpy array. We convert them to a list for replacement.
    water_data['Potability'] = water_data['Potability'].tolist()

    # Here we are replacing each data entry with either good or bad.
    for i in range(n):
        if water_data['Potability'][i] == 1:
            water_data['Potability'][i] = "bad"
        else:
            water_data['Potability'][i] = "good"

            # Thean again we are converting it to a dataframe.
    water_data = pd.DataFrame(water_data)
    return water_data


# Here data is split into training and testing data.
def split(X, y):
    X = preprocessing.scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # read the csv file and fill out the blank spaces with 0
    wine_data = pd.read_csv('winequality-white.csv', sep=';')
    water_data = pd.read_csv('water_potability.csv', sep=',')
    wine_data.fillna(0, inplace=True)
    water_data.fillna(0, inplace=True)

    wine_data = wine_quality(wine_data)
    water_data = water_potability(water_data)

    # selected features from the dataset
    # X = wine_data[['fixed acidity', 'citric acid', 'residual sugar', 'chlorides', 'total sulfur dioxide', 'pH', 'alcohol']]
    # y = wine_data[['quality']]
    X = water_data[['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Turbidity', 'Trihalomethanes']]
    y = water_data[['Potability']]
    y = np.array(y).reshape(len(y), )

    X_train, X_test, y_train, y_test = split(X, y)

    # Here we are using SVM classifier with Radial basis function kernel for classification.
    svc = svm.SVC()
    svc.fit(X_train, y_train)
    yhat = svc.predict(X_test)
    y_test = np.array(y_test)

    # Calculate the accuracy:
    count = 0
    for i in range(len(yhat)):
        if yhat[i] == y_test[i]:
            count += 1
    print('Water potability: ')
    print('Accuracy: ', count / len(y_test))
    # print('Wine quality: ')
    # print('Accuracy: ', count / len(y_test))