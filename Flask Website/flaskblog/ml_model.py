# data analysis and wrangling
import numpy as np

# to load model
import pickle
import os

LOAD_NAME = "current_model.sav"
LOAD_POS = "C:\\Users\\niv8s\\PycharmProjects\\Cyber\\Cyber Project\\Flask Website\\flaskblog\\machine_learning_models"


def load_model(file_name, save_path):
    """
    Loads a model from a specific path
    @param file_name: name of the model which the user wants to load
    @type file_name: string
    @param save_path: path of the model which the user wants to load
    @type save_path: string
    @return: the loaded model from the given path location
    @rtype: Any
    """
    full_name = os.path.join(save_path, file_name)
    loaded_model = pickle.load(open(full_name, 'rb'))
    return loaded_model


def model_pred(X_to_pred):
    """
    Predict a result for the given features
    @param model: a machine learning model
    @type model: Any
    @param X_to_pred: features of a user which wants to get evaluated
    @type X_to_pred: numpy array
    @return: the result predicted by the model
    @rtype: list
    """
    # to load a saved model
    loaded_model = load_model(LOAD_NAME, LOAD_POS)
    
    Y_pred = loaded_model.predict(X_to_pred)
    return Y_pred


def reshape_arr(arr):
    """
    Reshape an array into a numpy array that can be given to the model in order
    for it to return a prediction
    @param arr: an array of features
    @type arr: array
    @return: data that can be given to a model in order for it to predict a result
    @rtype: numpy array
    """
    # adding total symptoms column
    counter = 0
    for i in range(0, 5):
        counter += int(arr[i])
    arr = arr[:5] + [str(counter)] + arr[5:]

    numpy_data = np.array([arr], dtype='float64')
    numpy_data.reshape(1, -1)
    return numpy_data