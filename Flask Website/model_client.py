# data analysis and wrangling
import numpy as np

# to load model
import pickle
import os

from flask import Flask
from flaskblog.models import Features
import socket

ADDR = '127.0.0.1'
PORT = 1728
STOP_MESSAGE = 'end'

run = True


# create an array built of all features
def create_message(feat):
    """
    Create the message that will be sent to the server given the features
    @param feat: features data loaded from db
    @type feat: Features
    @return: the message created
    @rtype: string
    """
    message = ""
    for i in range(6):
        message += feat.symptoms[i]

    # to get if age is above or under 60
    message += (str(int(int(feat.age) / 60)))

    message += feat.gender
    message += feat.tested

    # adding total symptoms column
    counter = 0
    for i in range(0, 5):
        counter += int(message[i])
    message = message[:5] + str(counter) + message[5:]

    return message


def send_data(my_socket):
    """
    Sends the data to the server after loading all needed Features from the website's database.
    When finishes to send, will send a stop sign which will tell the server when to stop listening.
    @param my_socket: the socket which will be used to send the data through
    @type my_socket: socket
    """
    features = Features.query.all()
    if len(features) > 0:
        for feat in features:
            # filter rows that don't have features or a covid tests result
            if feat.tested is not None and feat.result is not None:
                message = create_message(feat)
                my_socket.send(message.encode())
                print(message)
    stop_sign = STOP_MESSAGE
    my_socket.send(stop_sign.encode())


def receive_model(my_socket):
    """
    Receives the Machine Learning model from the server and deserializes it
    @param my_socket: the socket which will be used to receive the data from
    @type my_socket: socket
    @return: the machine learning model which was sent from the server
    @rtype: Any
    """
    data = b""
    while True:
        packet = my_socket.recv(1024)
        if not packet:
            break
        data += packet
    model = pickle.loads(data)
    return model


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


def main():
    my_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    my_socket.connect((ADDR, PORT))

    send_data(my_socket)
    trained_model = receive_model(my_socket)
    print(trained_model)
    save_model('current_model.sav', 'C:\\Users\\niv8s\\PycharmProjects\\Cyber\\Cyber Project\\Flask Website\\flaskblog',
               trained_model)


if __name__ == "__main__":
    main()
