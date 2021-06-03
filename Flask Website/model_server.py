import os
import pickle
import socket
import pandas as pd 

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron

ADDR = '0.0.0.0'
PORT = 1728
STOP = 'end'
DF_PATH = 'C:\\Users\\niv8s\\PycharmProjects\\Cyber\\Cyber Project\\Model\\new_corona_df.csv'


# split the features from the message
def get_features(message):
    """
    Get a message and split it into an array with features
    @param message: a string containing features of a user
    @type message: string
    @return: an array containing all features
    @rtype: array
    """
    arr = []
    for i in range(len(message)):
        arr.append(int(message[i]))
    return arr


def receive_data(client_socket):
    """
    Receive all rows from the client's database and when given a specific string, will stop listening
    @return: all features of client's users
    @rtype: array
    """
    features_list = []
    message = client_socket.recv(1024).decode()
    # message = decrypt(message)
    while message != STOP:
        features = get_features(message)
        features_list.append(features)
        message = client_socket.recv(1024).decode()
        # message = decrypt(message)
    return features_list


# put data in matrix
def insert_data(data_arr):
    """
    Insert all given data into the server's current dataframe
    @param data_arr: the data which was sent from the client containing its user's features
    @type data_arr: arr
    @return: a dataframe which contains both old and new dataframes
    @rtype: pandas dataframe
    """
    new_df = pd.DataFrame.from_records(data_arr)
    # adding column name to the respective columns
    new_df.columns = ['Cough', 'Fever', 'Sore throat', 'Shortness of breath', 'Headache', 'Total symptoms', 'Contact',
                      'Above 60', 'Gender', 'Test result']

    old_df = pd.read_csv(DF_PATH)
    joined_df = pd.concat([old_df, new_df])

    return joined_df


# split list to X_train, Y_train
def split_data(df):
    """
    Split the given dataframe into a dataframe with the features, and a dataframe with the results
    @param df: a dataframe which will be split
    @type df: pandas dataframe
    @return: two split dataframes, one containing the features, and one containing the results
    @rtype: pandas dataframe
    """
    X_train = df.drop("Test result", axis=1)
    Y_train = df["Test result"]
    return X_train, Y_train


def load_model(filename, save_path):
    """

    @param filename:
    @type filename:
    @param save_path:
    @type save_path:
    @return:
    @rtype:
    """
    full_name = os.path.join(save_path, filename)
    loaded_model = pickle.load(open(full_name, 'rb'))
    return loaded_model


def train_model(X_train, Y_train):
    """

    @param X_train: 
    @type X_train:
    @param Y_train:
    @type Y_train:
    @return:
    @rtype:
    """
    # model = Perceptron(eta0=0.001, max_iter=10)
    model = GaussianNB()
    model.fit(X_train, Y_train)

    return model


# send the trained model back to client
def send_model(model, client_socket):
    pickled = pickle.dumps(model)
    client_socket.send(pickled)


def communication(client_socket):
    data = receive_data(client_socket)
    data_list = []
    for feat in data:
        arr = get_features(feat)
        data_list.append(arr)
    df = insert_data(data_list)
    X_train, Y_train = split_data(df)
    trained_model = train_model(X_train, Y_train)
    send_model(trained_model, client_socket)
    client_socket.close()


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((ADDR, PORT))
    server_socket.listen(1)
    while True:
        client_socket, address = server_socket.accept()
        communication(client_socket)


if __name__ == "__main__":
    main()

