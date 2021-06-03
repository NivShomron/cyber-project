# data analysis and wrangling
import os

# to save model
import pickle

# measure time
import time

# visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# grid search total epochs for the perceptron
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

# machine learning
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from xlwt import Workbook

# Need to add another column that shows how many symptoms the person has
# Use ROC to find the accuracy of the model

SAVE_NAME = 'perceptron_model.sav'
SAVE_PATH = "Cyber Project\\Model"

# create Workbook
wb = Workbook()


# Loading the data
def load_data(file_name, path):
    """
    Receives a specific path of a file and its name, loads the data into a pandas dataframe and returns it
    @param file_name: name of the file which you want to load
    @type file_name: string
    @param path: path of the file which you want to load
    @type path: string
    @return: returns the dataframe which was loaded from the given path
    @rtype: pandas dataframe
    """
    df = pd.read_csv(path + "\\" + file_name)
    return df


def split_data(df):
    """
    Receives a dataframe, splits it into train and test dataframes and returns them
    @param df: a dataframe which will be split
    @type df: pandas dataframe
    @return: two separate dataframes which will be used to train and test the machine learning models
    @rtype: pandas dataframe
    """
    train_df, test_df = train_test_split(df, test_size=0.2)
    return train_df, test_df


def clean_data(df):
    """
    receive dataframe, clean it and then save it
    @param df: a dataframe which will be cleaned
    @type df: pandas dataframe
    """
    # Changing names of columns
    df.rename(columns={"cough": "Cough", "fever": "Fever", "sore_throat": "Sore throat",
                       "shortness_of_breath": "Shortness of breath", "head_ache": "Headache",
                       "corona_result": "Test result", "age_60_and_above": "Above 60",
                       "gender": "Gender", "test_indication": "Contact"}, inplace=True)

    # Dropping columns that don't give useful info
    df = df.drop(columns=['test_date'])

    # Adding another column
    df['Total symptoms'] = df['Cough'] + df['Fever'] + df['Sore throat'] + df['Shortness of breath'] + df['Headache']
    cols = list(df.columns.values)

    # Changes the order of all the columns
    df = df[cols[0:5] + [cols[-1]] + [cols[-2]] + [cols[-4]] + [cols[-3]] + [cols[-5]]]
    cols = list(df.columns.values)

    nonestr = 'אחר'
    nullstr = 'NULL'

    # Remove rows containing missing data
    df = df[~df["Test result"].str.contains(nonestr, na=False)]
    df = df[~df["Above 60"].str.contains(nullstr, na=False)]
    df = df[~df["Gender"].str.contains(nullstr, na=False)]

    falsestr = 'שלילי'
    truestr = 'חיובי'
    femalestr = 'נקבה'
    malestr = 'זכר'
    below_60 = 'No'
    above_60 = 'Yes'
    otherstr = 'Other'
    abroadstr = 'Abroad'
    contactstr = 'Contact with confirmed'

    # Change text data to a binary value
    df["Test result"] = df["Test result"].replace(falsestr, 0)
    df["Test result"] = df["Test result"].replace(truestr, 1)

    df["Gender"] = df["Gender"].replace(femalestr, 0)
    df["Gender"] = df["Gender"].replace(malestr, 1)

    df["Above 60"] = df["Above 60"].replace(below_60, 0)
    df["Above 60"] = df["Above 60"].replace(above_60, 1)

    df["Contact"] = df["Contact"].replace(otherstr, 0)
    df["Contact"] = df["Contact"].replace(abroadstr, 0)
    df["Contact"] = df["Contact"].replace(contactstr, 1)

    test_val = [0, 1]
    gender_val = [0, 1]
    above_60 = [0, 1]
    contact_val = [0, 1]

    # Remove all non binary data left
    df = df[df['Test result'].isin(test_val)]
    df = df[df['Gender'].isin(gender_val)]
    df = df[df['Above 60'].isin(above_60)]
    df = df[df['Contact'].isin(contact_val)]

    # save changes to a new csv file
    df.to_csv('C:\\Users\\niv8s\\PycharmProjects\\Cyber\\Cyber Project\\Model\\new_corona_df.csv'
              , index=False, mode='a')


def compute_tp_tn_fn_fp(y_act, y_pred):
    """
    Receive a dataframe containing a model's prediction, and the actual result. will return true positive,
    true negative, false positive and false negative values.
    @param y_act: actual results of the covid tests
    @type y_act: pandas dataframe
    @param y_pred: prediction results of the machine learning model.
    @type y_pred: pandas dataframe
    @return: true positive, true negative, false positive and false negative values
    @rtype: int
    """
    tp = sum((y_act == 1) & (y_pred == 1))
    tn = sum((y_act == 0) & (y_pred == 0))
    fn = sum((y_act == 1) & (y_pred == 0))
    fp = sum((y_act == 0) & (y_pred == 1))
    return tp, tn, fp, fn


def compute_precision(tp, fp):
    """
    Computes the precision score when receiving true positive and false positive rates.
    @param tp: true positive rates
    @type tp: int
    @param fp: false positive rates
    @type fp: int
    @return: the calculated precision score
    @rtype: int
    """
    return (tp * 100) / float(tp + fp)


def compute_recall(tp, fn):
    """
    Computes the recall score when receiving true positive and false positive rates.
    @param tp: true positive rates
    @type tp: int
    @param fn: false negative rates
    @type fn: int
    @return: the calculated recall score
    @rtype: int
    """
    return (tp * 100) / float(tp + fn)


def compute_f1_score(y_act, y_pred):
    """
    Computes the f1 score when receiving model's prediction, and the actual result.
    @param y_act: dataframe containing the actual covid tests results
    @type y_act: pandas dataframe
    @param y_pred: dataframe containing the predicted covid test results by the model
    @type y_pred: pandas dataframe
    @return: the f1 calculated score
    @rtype: int
    """
    tp, tn, fp, fn = compute_tp_tn_fn_fp(y_act, y_pred)
    precision = compute_precision(tp, fp) / 100
    recall = compute_recall(tp, fn) / 100
    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score


def get_classifier_name(clf):
    """
    Finds the classifier name, when given a machine learning classifier
    @param clf: machine learning model classifier
    @type clf: Any
    @return: name of the classifier model
    @rtype: string
    """
    return str(type(clf)).split(".")[-1][:-2]


def compare_models(X_train, Y_train, X_test, Y_test):
    """
    Compares many different machine learning models, in order to find the one which fits the data the best.
    Saves the models' examination results in an excel file.
    @param X_train: dataframe containing all features, which are used to train the models
    @type X_train: pandas dataframe
    @param Y_train: dataframe containing all results, which are used to train the models
    @type Y_train: pandas dataframe
    @param X_test: dataframe containing all features, which are used to test the models
    @type X_test: pandas dataframe
    @param Y_test: dataframe containing all results, which are used to test the models
    @type Y_test: pandas dataframe
    """
    # creating the excel file
    sheet1 = wb.add_sheet('Sheet 1')

    # create the headlines
    sheet1.write(0, 1, 'Computation Time')
    sheet1.write(0, 2, 'Accuracy')
    sheet1.write(0, 3, 'Precision')
    sheet1.write(0, 4, 'Recall')
    sheet1.write(0, 5, 'F1 Score')
    sheet1.write(0, 6, 'AUC Score')

    # in order to write data in a different row each time
    row = 1

    prev_time = 0

    models = [LogisticRegression(), LinearSVC(), RandomForestClassifier(n_estimators=100), GaussianNB(), Perceptron(),
              SGDClassifier(), DecisionTreeClassifier(), GradientBoostingClassifier()]

    for clf in models:
        clf_name = get_classifier_name(clf)
        sheet1.write(row, 0, str(clf_name))

        time.process_time()

        model = clf
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        model.score(X_train, Y_train)
        acc = round(model.score(X_train, Y_train) * 100, 3)

        curr_time = time.process_time()
        proc_time = curr_time - prev_time
        sheet1.write(row, 1, round(proc_time, 3))
        prev_time = curr_time

        sheet1.write(row, 2, round(acc, 3))

        tp, tn, fn, fp = compute_tp_tn_fn_fp(Y_test, Y_pred)
        sheet1.write(row, 3, round(compute_precision(tp, fp), 3))
        sheet1.write(row, 4, round(compute_recall(tp, fn), 3))
        sheet1.write(row, 5, round(compute_f1_score(Y_test, Y_pred), 3))

        auc_score = roc_auc_score(Y_test, Y_pred)
        sheet1.write(row, 6, round(auc_score, 3))

        row += 1

    # saving the excel file with all the data
    wb.save("C:\\Users\\niv8s\\PycharmProjects\\Cyber\\Cyber Project\\Model\\model comparison.xls")


def train_model(X_train, Y_train):
    """
    Trains the model using X_train and Y_train dataframes
    @param X_train: dataframe containing all features which are used to train the model
    @type X_train: pandas dataframe
    @param Y_train: dataframe containing all results which are used to train the model
    @type Y_train: pandas dataframe
    @return: a trained machine learning model
    @rtype: Any
    """
    ml_model = Perceptron()
    ml_model.fit(X_train, Y_train)
    return ml_model


def load_model(filename, save_path):
    """
    Loads a model from a specific path
    @param filename: name of the model which the user wants to load
    @type filename: string
    @param save_path: path of the model which the user wants to load
    @type save_path: string
    @return: the loaded model from the given path location
    @rtype: Any
    """
    full_name = os.path.join(save_path, filename)
    loaded_model = pickle.load(open(full_name, 'rb'))
    return loaded_model


def save_model(filename, save_path, ml_model):
    '''
    Saves a model to a specific path
    @param filename: name of the file which the model will be saved as
    @type filename: string
    @param save_path: name of the path which the model will be saved at
    @type save_path: string
    @param ml_model: a machine learning model which will be saved in the given path
    @type ml_model: Any
    '''
    full_name = os.path.join(save_path, filename)
    pickle.dump(ml_model, open(full_name, 'wb'))


def model_pred(ml_model, X_test):
    '''
    Predicts a covid test result, when given the machine learning model and features
    @param ml_model: machine learning model which will be used to predict the covid test result
    @type ml_model: Any
    @param X_test: dataframe containing features which will be used to predict the result of the test
    @type X_test:
    @return: the covid test result predicted by the machine learning model
    @rtype: string array
    '''
    Y_pred = ml_model.predict(X_test)
    return Y_pred


def plot_features(ml_model, X_train):
    """
    Plot the importance of each feature of the pandas database (RandomForestClassifier)
    @param ml_model: the trained machine learning model
    @type ml_model: Any
    @param X_train: all features which were used to train the machine learning model
    @type X_train: pandas dataframe
    """
    feature_importance = ml_model.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(8, 18))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, X_train.keys()[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

"""
The numbers here represent the mean difference in the score (here: accuracy)
 the algorithm determined when the values of a particular feature are randomly shuffled
  before obtaining the score. So for example, a value of 0.20 means that shuffling this feature
   resulted in a drop of 0.20 in accuracy. Hence, this feature is very important.
"""


def plot_feature_importance(ml_model, X_test, Y_test):
    """
    Plot the importance of each feature and how impactful is it on evaluating a correct result (GaussianNB)
    @param ml_model: the trained machine learning model
    @type ml_model: Any
    @param X_test: all features which were used to train the model
    @type X_test: pandas dataframe
    @param Y_test: all covid test results which were used to train the model
    @type Y_test: pandas dataframe
    """
    imps = permutation_importance(ml_model, X_test, Y_test)

    imps_arr = list(imps.importances_mean)
    for item in imps_arr:
        item = str(round(float(item) * 1000, 3))

    _, ax = plt.subplots()

    ax.set_ylabel('Importance of feature')
    ax.set_title('Feature importance')

    ax.bar(features[0], imps_arr)
    ax.set_xticks(features[0])
    ax.set_yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])
    plt.show()


def best_learning_rate(model, X_train, Y_train):
    # define model evaluation method
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define grid
    grid = dict()
    grid['eta0'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
    # define search
    search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
    # perform the search
    results = search.fit(X_train, Y_train)
    # summarize
    print('Mean Accuracy: %.3f' % results.best_score_)
    print('Config: %s' % results.best_params_)
    # summarize all
    means = results.cv_results_['mean_test_score']
    params = results.cv_results_['params']
    for mean, param in zip(means, params):
        print(">%.3f with: %r" % (mean, param))

def best_total_epochs(model, X_train, Y_train):
    # define model evaluation method
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define grid
    grid = dict()
    grid['max_iter'] = [1, 10, 100, 1_000, 10_000]
    # define search
    search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
    # perform the search
    results = search.fit(X_train, Y_train)
    # summarize
    print('Mean Accuracy: %.3f' % results.best_score_)
    print('Config: %s' % results.best_params_)
    # summarize all
    means = results.cv_results_['mean_test_score']
    params = results.cv_results_['params']
    for mean, param in zip(means, params):
        print(">%.3f with: %r" % (mean, param))

# df = clean_data(df)
df = load_data('new_corona_df.csv', 'C:\\Users\\niv8s\\PycharmProjects\\Cyber\\Cyber Project\\Model')
train_df, test_df = split_data(df)

target = ['Test result']
features = [['Cough', 'Fever', 'Sore throat', 'Shortness of breath', 'Headache',
             'Total symptoms', 'Contact', 'Above 60', 'Gender']]

X_train = train_df.drop("Test result", axis=1)
Y_train = train_df["Test result"]
X_test = test_df.copy()
X_test = X_test.drop("Test result", axis=1)
Y_test = test_df["Test result"]

model = train_model(X_train, Y_train)

Y_test = test_df["Test result"]
Y_pred = model_pred(model, X_test)


# best_learning_rate(model, X_train, Y_train)
# best_total_epochs(model, X_train, Y_train)
# eta0: 0.001, max_iter: 10

# plot_feature_importance(model, X_test, Y_test)

# plot_features(model, X_train)
# buildROC(Y_test, Y_pred, model)