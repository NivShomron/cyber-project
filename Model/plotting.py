import pickle
import pandas as pd
import xlrd
import matplotlib.pyplot as plt
import numpy as np
import os

FOLDER_PATH = 'C:\\Users\\niv8s\\PycharmProjects\\Cyber\\Cyber Project\\Model'
EXCEL_NAME = 'model comparison.xls'
DF_NAME = 'new_corona_df.csv'

df = pd.read_csv(FOLDER_PATH + '\\' + DF_NAME)
excel_file = xlrd.open_workbook(FOLDER_PATH + '\\' + EXCEL_NAME)

sheet1 = excel_file.sheet_by_name('Sheet 1')


def plot_time_acc():
    """
    Plots a dotted graph showing the computational time and accuracy for each model
    """
    names = []
    times = []
    scores = []
    for row in range(8):
        name = sheet1.cell(row + 1, 0).value
        time = sheet1.cell(row + 1, 1).value
        score = sheet1.cell(row + 1, 5).value

        names.append(name)
        times.append(time)
        scores.append(score)

    _, ax = plt.subplots()
    ax.scatter(times, scores)

    for row, name_arr in enumerate(names):
        ax.annotate(name_arr, (times[row], scores[row]))


def plot_auc():
    """
    Plots a bars graph which shows the auc score of each Machine Learning model
    """
    names = []
    auc_scores = []

    for row in range(8):
        name = sheet1.cell(row + 1, 0).value
        score = sheet1.cell(row + 1, 6).value

        names.append(name)
        auc_scores.append(score)

    _, ax = plt.subplots()
    width = 0.4
    ax.bar(names, auc_scores, width)
    for i, (name, height) in enumerate(zip(names, auc_scores)):
        ax.text(i, height, ' ' + name, color='white',
                ha='center', va='top', rotation=-90, fontsize=15)
    ax.set_ylabel('AUC Score')
    ax.set_title('Models by AUC Score')
    ax.set_xticks([])  # remove the xticks, as the labels are now inside the bars


def plot_results(Y_train):
    """
    Plots a pie chart which shows the amount of Positive test results vs the Negative test results
    @param Y_train: the results of the training data covid tests
    @type Y_train: pandas dataframe
    """
    # Data to plot
    labels = 'Negative', 'Positive'
    sizes = [np.count_nonzero(Y_train == 0), np.count_nonzero(Y_train == 1)]
    explode = (0.1, 0)  # explode 1st slice
    # Plot
    plt.pie(sizes, explode=explode, labels=labels,
            autopct='%1.1f%%', shadow=True)
    plt.axis('equal')


Y_train = df["Test result"]

plt.show()
