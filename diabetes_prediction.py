import pandas
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

import numpy as np
from sklearn.svm import SVC

zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']


def replace_zero_values(data, column_name):
    """Method which helps replace 0 in column with its median value"""
    not_zero_values = data.loc[data[column_name] != 0, column_name]
    amount_for_replacing = len(data) - len(not_zero_values)
    # here was tested median and mean value. And median gave better results in all tests (2-4% better)
    median_value = not_zero_values.median()
    data.loc[data[column_name] == 0, column_name] = median_value
    print("Replaced %s values for '%s' column with median value of %s" % (amount_for_replacing, column_name, median_value))


def get_train_test_dataset(data, test_size=0.2):
    X = np.array(data.drop(['Outcome'], 1))  # filtering our features by removing outcome from dataset
    y = np.array(data['Outcome'])  # getting labels, its only 'Outcome' column

    # building training and testing subsets
    return train_test_split(X, y, test_size=test_size, random_state=100)


def study_different_models_and_data():
    datafile_name = 'data-files/Diabetes Dataset.csv'
    file_data = pandas.read_csv(datafile_name, encoding='latin-1')

    # Processing data in different ways. Base amount - 768
    # #1 - Filtering by removing 0 values for all columns except Age and Pregnancies
    filtered_zero_data = file_data[(file_data.BloodPressure != 0) & (file_data.SkinThickness != 0) & (file_data.Glucose != 0) &
                          (file_data.Insulin != 0) & (file_data.BMI != 0) & (file_data.DiabetesPedigreeFunction != 0) &
                          (file_data.Age != 0)]
    # Filtered amount = 392 records

    # #2 Replacing zeros in columns with its columns mean values
    mean_zero_data = file_data
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    for column_name in zero_columns:
        replace_zero_values(mean_zero_data, column_name)

    # Testing approcahes with different model algorithms and different data

    # KNN using filtered data
    train_X_filtered, test_X_filtered, train_y_filtered, test_y_filtered = get_train_test_dataset(filtered_zero_data)
    # using KNN classifier from sklearn library
    classifier = KNeighborsClassifier(n_neighbors=20)
    # training the classifier, based on our dataset
    classifier.fit(train_X_filtered, train_y_filtered)
    # checking the accuracy of classifier
    print('KNN filtered data. Accuracy: %s' % classifier.score(test_X_filtered, test_y_filtered))  # Show us Accuracy at ~72.15%

    # KNN using replaced zero data
    train_X_replaced, test_X_replaced, train_y_replaced, test_y_replaced = get_train_test_dataset(mean_zero_data)
    # using KNN classifier from sklearn library
    classifier = KNeighborsClassifier(n_neighbors=20)
    # training the classifier, based on our dataset
    classifier.fit(train_X_replaced, train_y_replaced)
    # checking the accuracy of classifier
    print('KNN replaced zero data. Accuracy: %s' % classifier.score(test_X_replaced, test_y_replaced))  # Show us Accuracy at ~75.97%

    # Here we can see that accuracy for dataset with replaced zero is better then filtered rows.

    # train_knn(train_X_replaced, test_X_replaced, train_y_replaced, test_y_replaced)

    # using Support Vector classifier from sklearn library with replaced zero, tweaked by 'linear' kernel.
    classifier = SVC(kernel='linear')
    # training the classifier, based on our dataset
    classifier.fit(train_X_filtered, train_y_filtered)
    # checking the accuracy of classifier
    print('SVC filtered data. Accuracy: %s' % classifier.score(test_X_filtered, test_y_filtered))
    # Default accuracy: 68.35%. Tweaked: 79.74%

    # using Support Vector classifier from sklearn library with filtered zero, tweaked by 'linear' kernel.
    # Also tweaked method takes more time then the default one. but we need better results, rather than perfomance.
    classifier = SVC(kernel='linear')
    # training the classifier, based on our dataset
    classifier.fit(train_X_replaced, train_y_replaced)
    # checking the accuracy of classifier
    print('KNN replaced data. Accuracy: %s' % classifier.score(test_X_replaced, test_y_replaced))
    # Default accuracy: 65.58%. Tweaked: 73.37%

    # Running through few more models to see any other results.

    classifier = GradientBoostingClassifier()
    classifier.fit(train_X_filtered, train_y_filtered)
    # checking the accuracy of classifier
    print('GBC Accuracy for filtered data: %s' % classifier.score(test_X_filtered, test_y_filtered))
    # Accuracy: 79.74%

    classifier = GradientBoostingClassifier()
    classifier.fit(train_X_replaced, train_y_replaced)
    # checking the accuracy of classifier
    print('GBC Accuracy for replaced data: %s' % classifier.score(test_X_replaced, test_y_replaced))
    # Accuracy: 72.07%

    classifier = LogisticRegression()
    classifier.fit(train_X_filtered, train_y_filtered)
    # checking the accuracy of classifier
    print('LR Accuracy for filtered data: %s' % classifier.score(test_X_filtered, test_y_filtered))
    # Accuracy: 74.68%

    classifier = LogisticRegression()
    classifier.fit(train_X_replaced, train_y_replaced)
    # checking the accuracy of classifier
    print('LR Accuracy for replaced data: %s' % classifier.score(test_X_replaced, test_y_replaced))
    # Accuracy: 74.67%


def train_knn(train_X, test_X, train_y, test_y):
    """
    This part was used to train KNN by selecting most optimal neighbours amount. Optimal value was 20.
    (optimal_neighborhood.png)
    """
    indexes = range(2, 100)
    values = []
    for k in range(2, 100):
        classifier = KNeighborsClassifier(n_neighbors=k)
        # training the classifier, based on our dataset
        classifier.fit(train_X, train_y)
        values.append(classifier.score(test_X, test_y))
        print(classifier.score(test_X, test_y), k)

    plt.plot(indexes, values, label='Show accuracy with different K')
    plt.show()


def predict_diabet(pregnancies, glucose, blood_pressure, skin, insulin, bmi, dpf, age):
    """Predicting weather diabetes or not based on values using KNN algorithm for replaced data and using"""
    datafile_name = 'data-files/Diabetes Dataset.csv'
    file_data = pandas.read_csv(datafile_name, encoding='latin-1')
    mean_zero_data = file_data

    for column_name in zero_columns:
        replace_zero_values(mean_zero_data, column_name)

    # KNN using replaced zero data
    train_X_replaced, test_X_replaced, train_y_replaced, test_y_replaced = get_train_test_dataset(mean_zero_data)
    # using KNN classifier from sklearn library
    classifier_knn = KNeighborsClassifier(n_neighbors=20)
    # training the classifier, based on our dataset
    classifier_knn.fit(train_X_replaced, train_y_replaced)
    score_knn = round(classifier_knn.score(test_X_replaced, test_y_replaced) * 100, 2)
    knn_mean_zero_result = classifier_knn.predict([[pregnancies, glucose, blood_pressure, skin, insulin, bmi, dpf, age]])[0]
    knn_diabetes = 'Positive' if knn_mean_zero_result else 'Negative'

    # using Support Vector classifier from sklearn library with replaced zero, tweaked by 'linear' kernel.
    filtered_zero_data = file_data[(file_data.BloodPressure != 0) & (file_data.SkinThickness != 0) & (file_data.Glucose != 0) &
                          (file_data.Insulin != 0) & (file_data.BMI != 0) & (file_data.DiabetesPedigreeFunction != 0) &
                          (file_data.Age != 0)]
    train_X_filtered, test_X_filtered, train_y_filtered, test_y_filtered = get_train_test_dataset(filtered_zero_data)
    classifier_svc = SVC(kernel='linear')
    # training the classifier, based on our dataset
    classifier_svc.fit(train_X_filtered, train_y_filtered)
    score_svc = round(classifier_svc.score(test_X_filtered, test_y_filtered) * 100, 2)
    svc_filtered_zero_result = classifier_svc.predict([[pregnancies, glucose, blood_pressure, skin, insulin, bmi, dpf, age]])
    svc_diabetes = 'Positive' if svc_filtered_zero_result else 'Negative'

    print('Using data with replaced zeros with median value, it is %s percent that Diabetes is %s' % (score_knn, knn_diabetes))
    print('Using full real data, it is %s percent that Diabetes is %s' % (score_svc, svc_diabetes))


pregnancies = int(input('Please provide amount of Pregnancies: '))
glucose = int(input('Please provide Glucose level: '))
blood_pressure = int(input('Please provide Blood Pressure: '))
skin = int(input('Please provide thickness of skin: '))
insulin = int(input('Please provide Insulin level: '))
bmi = float(input('Please provide BMI: '))
dpf = float(input('Please provide Diabetes Pedigree Function: '))
age = int(input('Please provide Age: '))
predict_diabet(pregnancies, glucose, blood_pressure, skin, insulin, bmi, dpf, age)