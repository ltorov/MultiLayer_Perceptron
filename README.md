# MultiLayer_Perceptron
This Python script includes functions for normalizing and sampling datasets, along with the design and training of a neural network using a multilayer perceptron (MLP). The main functionalities are outlined below:

normalize(dataset)

This function loads a dataset from a MATLAB .mat file and performs data normalization. It extracts specific variables, combines them to create input-output pairs, calculates minimum and maximum values across datasets, and returns both the original and normalized input-output pairs.

sampling(X, Y, plot=False)

This function splits and samples the dataset into training, testing, and validation sets. It partitions the input dataset (X, Y) into three subsets based on specified proportions (75%, 25%, 0%). It also provides the option to create visualizations of the partitioned data.

MLP class:

__init__(number_of_inputs, input_nodes, hidden_layers, output_nodes, activation_functions)

Initialize a Multilayer Perceptron (MLP) instance. It takes the number of inputs, the number of nodes in the input layer, a list representing the number of nodes in each hidden layer, the number of nodes in the output layer, and a list of activation function names for each layer. The function initializes instance variables, calculates the number of layers, and initializes random weights.

linear(x)

Linear activation function. Takes input 'x' and returns the output of the linear activation function.

sigmoid(x)

Sigmoid activation function. Takes input 'x' and returns the output of the sigmoid activation function.

tanh(x)

Hyperbolic tangent (tanh) activation function. Takes input 'x' and returns the output of the tanh activation function.

activation_function(v, phi, a=1, b=0)

Compute the output of the specified activation function. Takes the input to the activation function ('v'), the name of the activation function ('phi'), and optional parameters 'a' and 'b'. Returns the output of the specified activation function.

derivative_activation_function(v, phi, a=1)

Compute the derivative of an activation function. Takes the input to the activation function ('v'), the name of the activation function ('phi'), and optional scaling factor 'a'. Returns the derivative of the activation function applied to 'v'.

forward(x)

Perform a forward pass through the Multilayer Perceptron (MLP). Takes input data for the forward pass, calculates activation outputs for all layers, and returns the final activation output, list of activation outputs for all layers, and list of derivatives of activation functions for all layers.

backpropagation(y_desired, out, Phi, Phi_prime, Gradients_dic, learning_rate, method, error=None)

Perform backpropagation to update the network's weights. Takes the desired output, forward pass output, activation outputs, derivatives of activation functions, dictionary to store gradients, learning rate, training method, and optional error value for batch training. Returns the calculated error during backpropagation, list of gradients for each layer, updated weights, and gradients dictionary.

sequential(X, Y_desired, learning_rate, show)

Perform sequential training for a neural network. Takes input data and desired output for training, learning rate, and optional parameter to display debugging information. Returns a dictionary containing gradients for each layer sorted by layer index.

batch(X, Y_desired, learning_rate, show)

Perform batch training for a neural network. Takes input data and desired output for training, learning rate, and optional parameter to display debugging information. Returns a dictionary containing gradients for each layer sorted by layer index.

epoch(X, Y_desired, Ne, method, learning_rate, show=False, stop_criteria=False)

Perform training for a neural network over multiple epochs. Takes input data, desired output, number of epochs, training method, learning rate, and optional parameters to display debugging information and enable a stop criteria. Returns a dictionary containing gradients for each layer at each epoch and the average energy error.

train(T, YT, Ne, method, model, learning_rate, show=False, plot=True)

Train a neural network using the specified training method and parameters. Takes training data, desired output, number of epochs, training method, model identifier, learning rate, and optional parameters to display debugging information and plot gradients. Returns a dictionary containing gradients for each layer at each epoch and the average energy error.

test(T, YT, t, Yt, model, tolerance=1e-2, show=False, plot=True)

Test a trained neural network's performance on a dataset. Takes training data, desired output for training, testing data, desired output for testing, model identifier, tolerance level for errors, and optional parameters to display test results and plot. Returns the percentage of data points with errors exceeding the tolerance and the average energy error.

iteration(X, Y, Ne, method, model, tolerance=1e-2, max_iter=50, learning_rate=1, show=False, plot=True)

Perform iterative training with validation. Takes combined dataset, number of epochs, training method, model identifier, tolerance level for errors, maximum iterations, learning rate, and optional parameters to display results and plot. Returns the percentage of data points with errors exceeding the tolerance, validation dataset, and predicted output for the validation dataset.

validation(V, YV, model, tolerance=1e-2, show=False, plot=True)

Validate a trained neural network's performance on a validation dataset. Takes validation data, desired output for validation, model identifier, tolerance level for errors, and optional parameters to display validation results and plot. Returns the percentage of data points with errors exceeding the tolerance.


Main Section

The script includes a main section that:

Creates a virtual environment.
Activates the virtual environment.
Installs packages listed in 'requirements.txt'.
Designs the neural network architecture with specified parameters.
Reads the dataset and normalizes it.
Trains two models with different configurations using the MLP class.
Usage

Ensure that you have a MATLAB .mat file with the required variables. Update the dataset path accordingly. The script creates and activates a virtual environment, installs necessary packages, and trains two neural network models. Adjust the neural network parameters, such as learning rate, hidden layers, and activation functions, to fit your specific use case.

Note: The script uses a custom MLP class, so make sure the associated source code (multilayer_perceptron.py) is present in the 'src' directory.

Requirements

Make sure to have the necessary packages listed in 'requirements.txt' installed in your virtual environment.
