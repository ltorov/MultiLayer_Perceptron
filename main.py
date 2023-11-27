import subprocess
from src.multilayer_perceptron import MLP

def normalize(dataset):
    """
    Normalize a dataset stored in a MATLAB .mat file.

    This function loads a dataset from a .mat file and performs data normalization. It extracts the 'COD1', 'Xv1',
    and 't' variables from the file, combines 'COD1' and 'Xv1' to create the input data 'X', and uses 't' as the
    desired output 'Y'. The function then calculates the minimum and maximum values across all datasets, normalizes
    the data, and returns both the original and normalized input-output pairs.

    Parameters:
    - dataset (str): Path to the .mat file containing the dataset.

    Returns:
    - numpy.ndarray: Original input data 'X' (unnormalized).
    - numpy.ndarray: Original desired output data 'Y' (unnormalized).
    - numpy.ndarray: Normalized input data 'X_norm'.
    - numpy.ndarray: Normalized desired output data 'Y_norm'.
    """
    # Load the .mat file
    data = scipy.io.loadmat(dataset)

    # Access the 'COD1' variable data
    cod1 = data['COD1']
    # Access the 'Xv1' variable data
    Xv1 = data['Xv1']
    # Access the 't' variable data
    t = data['t']

    # Create the X array by combining cod1 and Xv1
    X = np.column_stack((cod1, Xv1))

    # Create the Y desired array with t
    Y = t

    XY = np.column_stack((X,Y))

    # Find the maximum and minimum values across all three datasets
    max = np.max(XY, axis = 0)
    min = np.min(XY, axis = 0)
    XY_norm = (XY - min) / (max - min)

    # Create the X array by combining normalized cod1 and Xv1
    X_norm = XY_norm[:, :2]

    # Create the Y desired array with normalized t
    Y_norm = XY_norm[:, -1][:, np.newaxis]

    return X, Y, X_norm, Y_norm

def sampling(X, Y, plot=False):
    """
    Split and sample the dataset into training, testing, and validation sets.

    This function partitions the input dataset (X, Y) into three subsets: training, testing, and validation.
    The sizes of the subsets are determined by the specified proportions (60%, 20%, 20%). It also provides
    the option to create visualizations of the partitioned data if required.

    Parameters:
    - X (numpy.ndarray): Input data samples.
    - Y (numpy.ndarray): Desired output data corresponding to the input samples.
    - plot (bool): If True, generate visualizations of the partitioned data.

    Returns:
    - numpy.ndarray: Training input data.
    - numpy.ndarray: Training desired output data.
    - numpy.ndarray: Testing input data.
    - numpy.ndarray: Testing desired output data.
    - numpy.ndarray: Validation input data.
    - numpy.ndarray: Validation desired output data.
    - numpy.ndarray: Indexes of the training data points.
    - numpy.ndarray: Indexes of the testing data points.
    - numpy.ndarray: Indexes of the validation data points.
    """
    lenT, lent, lenV = round(len(X)*0.6), round(len(X)*0.2), round(len(X)*0.2)

    # Shuffle indexes to ensure uniqueness
    all_indexes = np.arange(len(X))
    np.random.shuffle(all_indexes)

    indT = all_indexes[:lenT]
    indt = all_indexes[lenT:lenT+lent]
    indV = all_indexes[lenT+lent:lenT+lent+lenV]

    T = X[indT]; YT = Y[indT]
    t = X[indt]; Yt = Y[indt]
    V = X[indV]; YV = Y[indV]

    if plot:
        _, axs = plt.subplots(1, 3, figsize=(15, 5))
        ax = axs[0]
        ax.plot(Y, X[:,0],linestyle='-', color = 'darkgrey')
        ax.scatter(YT, T[:,0],marker='*', color = 'black')
        ax.plot(Y, X[:,1],linestyle='-', color = 'darkgrey')
        ax.scatter(YT, T[:,1],marker='v', color = 'black')
        ax.set_title('Training set')
        ax.set_xlabel('t')
        ax.legend(['COD1', 'Sampled COD1','Xv1', 'Sampled Xv1'])

        ax = axs[1]
        ax.plot(Y, X[:,0],linestyle='-', color = 'darkgrey')
        ax.scatter(Yt, t[:,0],marker='*', color = 'black')
        ax.plot(Y, X[:,1],linestyle='-', color = 'darkgrey')
        ax.scatter(Yt, t[:,1],marker='v', color = 'black')
        ax.set_title('Testing set')
        ax.set_xlabel('t')
        ax.legend(['COD1', 'Sampled COD1','Xv1', 'Sampled Xv1'])

        ax = axs[2]
        ax.plot(Y, X[:,0],linestyle='-', color = 'darkgrey')
        ax.scatter(YV, V[:,0],marker='*', color = 'black')
        ax.plot(Y, X[:,1],linestyle='-', color = 'darkgrey')
        ax.scatter(YV, V[:,1],marker='v', color = 'black')
        ax.set_title('Validation set')
        ax.set_xlabel('t')
        ax.legend(['COD1', 'Sampled COD1','Xv1', 'Sampled Xv1'])

        plt.tight_layout()
        plt.suptitle("Data Partitioning and Sampling")
        plt.subplots_adjust(top=0.8) 
        plt.savefig('data/Data Partitioning and Sampling')
        plt.show()

    return T, YT, t, Yt, V, YV, indT, indt, indV

if __name__ == "__main__":
    # System imports
    from venv import create
    from os.path import join, expanduser, abspath
    from subprocess import run
    import os

    # Create virtual environment
    try:
        dir = join(expanduser("."), "venv")
        create(dir, with_pip=True)
        print("Virtual environment created on: ", dir)
    except Exception as e:
        raise Exception("Failed to create the virtual environment: " + str(e))

    # Activate virtual environment
    activate_script = join(dir, "bin", "activate")
    try:
        activate_cmd = f"source {activate_script}" if os.name != "nt" else activate_script
        run(activate_cmd, shell=True, check=True)
    except Exception as e:
        raise Exception("Failed to activate the virtual environment: " + str(e))


    # Install packages in 'requirements.txt'.
    try:
        run(["python3", "-m", "pip3", "install", "--upgrade", "pip3"])
        run(["bin/pip3", "install", "-r", abspath("requirements.txt")], cwd=dir)
    except:
        run(["python", "-m", "pip", "install", "--upgrade", "pip"])
        run(["bin/pip", "install", "-r", abspath("requirements.txt")], cwd=dir)
    finally:
        print("Completed installation of requirements.")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import scipy.io
    # Design neural network arquitecture
    number_of_inputs = 2
    input_nodes = 2
    hidden_layers = [2]
    output_nodes = 1

    # Select activation function for all layers
    activation_functions = ["sigmoid", "sigmoid", "sigmoid"]

    # Determine number of epochs per sequence
    Ne = 50
    
    # Set tolerance for error
    tolerance = 1*10**(-2)

    # Read the dataset
    _,_, X_norm, Y_norm_desired = normalize('data/datos.mat')

    # Worst model
    learning_rate = 0.9
    hidden_layers = [3,3]
    activation_functions = ["sigmoid", "sigmoid", "sigmoid", "sigmoid"]
    model = "eta 0.9 L=2 l=3"
    T, YT, t, Yt, V, YV, _,_,_ = sampling(X_norm, Y_norm_desired, plot = False)
    mlp = MLP(number_of_inputs, input_nodes, hidden_layers, output_nodes, activation_functions)
    delta, V, YV = mlp.iteration(X_norm, Y_norm_desired, Ne, 'Batch', model, max_iter = 10, learning_rate = learning_rate, show = False, plot = True)
    epsilon = mlp.validation(V, YV, model, plot = True)

    # Median model
    learning_rate = 0.5
    hidden_layers = [4]
    activation_functions = ["sigmoid", "sigmoid", "sigmoid"]
    model = "eta 0.5 L=1 l=4"
    T, YT, t, Yt, V, YV, _,_,_ = sampling(X_norm, Y_norm_desired, plot = False)
    mlp = MLP(number_of_inputs, input_nodes, hidden_layers, output_nodes, activation_functions)
    delta, V, YV = mlp.iteration(X_norm, Y_norm_desired, Ne, 'Batch', model, max_iter = 10, learning_rate = learning_rate, show = False, plot = True)
    epsilon = mlp.validation(V, YV, model, plot = True)

