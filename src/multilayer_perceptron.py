import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, number_of_inputs, input_nodes, hidden_layers, output_nodes, activation_functions):
        """
        Initialize a Multilayer Perceptron (MLP) instance.

        Parameters:
        - number_of_inputs (int) : Number of inputs.
        - input_layer (int): Number of input nodes.
        - hidden_layers (list): List of integers representing the number of nodes in each hidden layer.
        - output_layer (int): Number of output nodes.
        - activation_functions (list): List of activation function names for each layer.

        Initializes instance variables, calculates the number of layers, and initializes weights.
        """
        self.number_of_inputs = number_of_inputs
        self.input_nodes = input_nodes
        self.hidden_layers = hidden_layers
        self.output_nodes = output_nodes
        self.activation_functions = activation_functions
        self.num_layers = len(hidden_layers) + 2
        self.weights = []

        # Initialize random weights for all layers
        layer_sizes = [number_of_inputs] + [input_nodes] + hidden_layers + [output_nodes]
        for i in range(1, len(layer_sizes)):
            self.weights.append(np.random.rand(layer_sizes[i],layer_sizes[i-1]))
            # Initialize weights of 1 for all layers for testing
            # self.weights.append(np.ones((layer_sizes[i], layer_sizes[i-1])))

    def linear(self, x):
        """
        Linear activation function.

        Parameters:
        - x (numpy.ndarray): Input to the activation function.

        Returns:
        - numpy.ndarray: Output of the linear activation function.
        """
        return x

    def sigmoid(self, x):
        """
        Sigmoid activation function.

        Parameters:
        - x (numpy.ndarray): Input to the activation function.

        Returns:
        - numpy.ndarray: Output of the sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        """
        Hyperbolic tangent (tanh) activation function.

        Parameters:
        - x (numpy.ndarray): Input to the activation function.

        Returns:
        - numpy.ndarray: Output of the tanh activation function.
        """
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def activation_function(self, v, phi, a =1, b=0):
        """
        Compute the output of the specified activation function.

        Parameters:
        - v (numpy.ndarray): The input to the activation function (local induced field).
        - phi (str): The name of the activation function to be applied.
        - a (float, optional): The 'a' parameter for the activation function (default is 1).
        - b (float, optional): The 'b' parameter for the activation function (default is 0).

        Returns:
        - numpy.ndarray: The output of the specified activation function.
        
        Raises:
        - ValueError: If an unsupported activation function is provided.
        """
        if phi == "linear": return a*v + b
        elif phi == "sigmoid": return 1 / (1 + np.exp(-a*v))
        elif phi == "tanh": return (np.exp(a*v) - np.exp(-a*v)) / (np.exp(a*v) + np.exp(-a*v))
        else: raise ValueError(f"Unsupported activation function: {phi}") 
        
    def derivative_activation_function(self, v, phi, a=1):
        """
        Compute the derivative of an activation function.

        Parameters:
        - v (numpy.ndarray): The input to the activation function (local induced field).
        - phi (str): The name of the activation function.
        - a (float, optional): Scaling factor for the activation function. Default is 1.

        Returns:
        - numpy.ndarray: The derivative of the activation function applied to v.

        Raises:
        - ValueError: If an unsupported activation function is provided.
        """
        if phi == "linear": return a*np.ones(v.shape)
        elif phi == "sigmoid": return a*(np.exp(-a*v)/(1+np.exp(-a*v))**2)
        #elif phi == "tanh": return a*(1 - ((np.exp(a*v) - np.exp(-a*v)) / ((np.exp(a*v) + np.exp(-a*v))**2)))
        elif phi == "tanh": return a * (1 - (np.tanh(a*v)**2))
        else: raise ValueError(f"Unsupported activation function: {phi}") 
    
    def forward(self, x):
        """
        Perform a forward pass through the Multilayer Perceptron (MLP).

        Parameters:
        - x (numpy.ndarray): Input data for the forward pass (one row).

        Returns:
        - numpy.ndarray: Output of the forward pass.
        - list: List of activation outputs for all layers.
        - list: List of derivatives of activation functions for all layers.
        """
        # Initialize lists to store activation outputs (yi) and their derivatives (yi_prime)
        Phi = []; Phi_prime = [] 
        Phi.append(x.reshape(-1,1))

        # Transpose the input data for compatibility with weight matrix dimensions
        x = np.transpose(x)

        # Calculate the local induced field (vi) for the first hidden layer
        vi = np.dot(self.weights[0], x)

        # Apply the activation function to compute the activation output (yi)
        yi = self.activation_function(vi, self.activation_functions[0])

        # Calculate the derivative of the activation function for the output layer (yi_prime)
        yi_prime = self.derivative_activation_function(vi, self.activation_functions[0])

        # Append the calculated activation output and its derivative to the lists
        Phi.append(yi.reshape(-1,1)); Phi_prime.append(yi_prime.reshape(-1,1))

        # Loop through hidden layers and output layer to compute activations and derivatives
        for i in range(1, self.num_layers):
            # Calculate the local induced field (vi) for the current layer
            vi = np.dot(self.weights[i], np.transpose(yi))
            
            # Apply the activation function to compute the activation output (yi)
            yi = np.array(self.activation_function(vi, self.activation_functions[i]))
            
            # Calculate the derivative of the activation function for the current layer (yi_prime)
            yi_prime = np.array(self.derivative_activation_function(vi, self.activation_functions[i]))

            # Append the calculated activation output and its derivative to the lists
            Phi.append(yi.reshape(-1,1)); Phi_prime.append(yi_prime.reshape(-1,1))

        # Return the final activation output (yi), list of activation outputs (Phi), and list of derivatives (Phi_prime)
        return yi, Phi, Phi_prime

    def backpropagation(self, y_desired, out, Phi, Phi_prime, Gradients_dic, learning_rate, method, error=None):
        """
        Perform backpropagation to update the network's weights.

        Parameters:
        - y_desired (numpy.ndarray): Desired output for the given input data.
        - out (numpy.ndarray): Output of the forward pass.
        - Phi (list): List of activation outputs for each layer during the forward pass.
        - Phi_prime (list): List of derivatives of activation outputs for each layer.
        - Gradients_dic (dict): Dictionary to store gradients for each layer and neuron.
        - learning_rate (float): Learning rate used for weight updates during backpropagation.
        - method (str): The training method used ('Sequential' or 'Batch').
        - error (float): Error value, if provided (used for batch training).

        Returns:
        - float: Calculated error during backpropagation.
        - list: List of calculated gradients for each layer.
        - dict: Updated weights for each layer after backpropagation.
        """

        # Initialize a list to store gradients for each layer
        Gradients = []

        # Calculate the total error between desired output and actual output if method is 'Sequential'
        if method == 'Sequential':
            error = np.sum(y_desired - out)

        # Calculate the delta for the last layer
        # This is the error scaled by the derivative of the last layer's activation function
        delta_k = -error * Phi_prime[-1]

        Gradients.append(delta_k)  # Store the gradient for the last layer

        key_layer = 'Layer ' + str(self.num_layers)
        if key_layer not in Gradients_dic:
            Gradients_dic[key_layer] = {}

        for index, delta in enumerate(delta_k):
            key_neuron = 'Neuron ' + str(index + 1)
            if key_neuron not in Gradients_dic[key_layer]:
                Gradients_dic[key_layer][key_neuron] = []
            Gradients_dic[key_layer][key_neuron].append(delta[0])

        # Update the weights for the output layer using delta_k and input from the second-to-last layer
        self.weights[-1] += learning_rate * np.dot(delta_k, Phi[-2].T)

        # Initialize delta_m with delta_k for the upcoming calculations
        delta_m = delta_k

        # Loop through hidden layers and update weights and deltas
        for layer in range(1, self.num_layers):
            # Calculate delta_m for the current hidden layer
            # This delta_m is the product of the previous delta_m and the derivative of the current layer's activation function
            delta_m = np.multiply(Phi_prime[-(layer + 1)], np.dot(self.weights[-layer].T, delta_m))

            # Update the weights for the current hidden layer using delta_m and input from the previous layer
            self.weights[-(layer + 1)] += learning_rate * np.dot(delta_m, Phi[-(layer + 2)].T)

            Gradients.append(delta_m)

            key_layer = 'Layer ' + str(self.num_layers - layer)

            if key_layer not in Gradients_dic:
                Gradients_dic[key_layer] = {}
            for index, delta in enumerate(delta_m):
                key_neuron = 'Neuron ' + str(index + 1)
                if key_neuron not in Gradients_dic[key_layer]:
                    Gradients_dic[key_layer][key_neuron] = []
                Gradients_dic[key_layer][key_neuron].append(delta[0])

        # Return the total error, list of gradients, and updated weights
        return error, Gradients, Gradients_dic, self.weights

    def sequential(self, X, Y_desired, learning_rate, show):
        """
        Perform sequential training for a neural network.

        This function iteratively processes each input-output pair (X, Y_desired) and performs a forward pass
        followed by backpropagation to update the network's weights.

        Parameters:
        - X (list): List of input data samples.
        - Y_desired (list): List of desired output data corresponding to the input samples.
        - learning_rate (float): Learning rate used for weight updates during backpropagation.
        - show (bool): If True, display debugging information for each step.

        Returns:
        - dict: A dictionary containing gradients for each layer sorted by layer index.
        """

        Gradients_dic = {}; Errors = np.zeros(len(X))
        method = 'Sequential'

        # Iterate over input-output pairs
        for index, (x, y_desired) in enumerate(zip(X, Y_desired)):
            out, Phi, Phi_prime = self.forward(x)
            error = np.sum(y_desired - out)
            Errors[index] = error
            
            error, Gradients, Gradients_dic, Weights = self.backpropagation(y_desired, out, Phi, Phi_prime, Gradients_dic, learning_rate, method)

            if show:
                print("----------------------Point ", index+1, "----------------------")
                print('x', x)
                print('y desired', y_desired)

                print("----------------------Forward Pass----------------------")
                print("Forward output: ", out)
                print("Phi: ", Phi)
                print("Phi prime: ", Phi_prime)

                print("----------------------Backpropagation----------------------")
                print('Error', error)
                print('Gradients', Gradients)
                print('Weights', Weights)
        average_energy_error = np.mean([(x**2)/2 for x in Errors])
        return {key: Gradients_dic[key] for key in sorted(Gradients_dic)}, average_energy_error
    
    def batch(self, X, Y_desired, learning_rate, show):
        """
        Perform batch training for a neural network.

        This function processes the entire dataset (X, Y_desired) as a batch. It computes the forward pass
        for each data point, accumulates the errors, and performs backpropagation using the point with the
        highest error. The gradients for each layer are collected and returned.

        Parameters:
        - X (list): List of input data samples.
        - Y_desired (list): List of desired output data corresponding to the input samples.
        - learning_rate (float): Learning rate used for weight updates during backpropagation.
        - show (bool): If True, display debugging information.

        Returns:
        - dict: A dictionary containing gradients for each layer sorted by layer index.
        """

        Gradients_dic = {}; Errors = np.zeros(len(X)); Phis = []

        # Process each data point
        for index, (x, y_desired) in enumerate(zip(X, Y_desired)):
            out, Phi, Phi_prime = self.forward(x)
            error = np.sum(y_desired - out)
            Errors[index] = error
            Phis.append( [Phi, Phi_prime])

            if show:
                print("----------------------Point ", index+1, "----------------------")
                print('x: ', x)
                print('y desired: ', y_desired)

                print("----------------------Forward Pass----------------------")
                print("Forward output: ", out)
                print("Phi: ", Phi)
                print("Phi prime: ", Phi_prime)
        
        average_energy_error = np.mean([(x**2)/2 for x in Errors])
        max_index = Errors.argmax()
        phis = Phis[max_index] 
        Phi = phis[0]; Phi_prime = phis[1]
        x = X[max_index]; y_desired = Y_desired[max_index]
        error, Gradients, Gradients_dic, Weights = self.backpropagation(y_desired, out, Phi, Phi_prime, Gradients_dic, learning_rate, 'Batch', np.mean(Errors))

        if show:
            print("----------------------Backpropagation----------------------")
            print('Last error: ', error)
            print('Gradients: ', Gradients)
            print('Weights: ', Weights)
            print('Mean Error: ', np.mean(Errors))

        return {key: Gradients_dic[key] for key in sorted(Gradients_dic)}, average_energy_error
    
    def epoch(self, X, Y_desired, Ne, method, learning_rate, show = False, stop_criteria = False):
        """
        Perform training for a neural network over multiple epochs.

        This method iterates over a specified number of epochs, where each epoch consists of processing
        the entire dataset (X, Y_desired) using either the 'sequential' method or the 'batch' method,
        based on the specified 'method' parameter. It collects and returns the gradients for each layer
        at each epoch.

        Parameters:
        - X (list): List of input data samples.
        - Y_desired (list): List of desired output data corresponding to the input samples.
        - Ne (int): Number of epochs to run the training.
        - method (str): The training method to use, either 'Sequential' or 'Batch'.
        - learning_rate (float): Learning rate used for weight updates during backpropagation.
        - show (bool): If True, display debugging information for each epoch.

        Returns:
        - dict: A dictionary containing gradients for each layer at each epoch, indexed by the epoch number.
        """

        Gradient_Matrix = {}
        for epoch in range(Ne):
            if show:
                print("----------------------Epoch ", epoch+1, "----------------------")
            if method == "Sequential":
                Gradient_Matrix['Epoch '+str(epoch+1)], average_energy_error = self.sequential(X, Y_desired, learning_rate, show)
            elif method == "Batch":
                Gradient_Matrix['Epoch '+str(epoch+1)], average_energy_error = self.batch(X, Y_desired, learning_rate, show)
            
                if stop_criteria:
                    # Iterate through epochs, layers, and nodes
                    for epoch, epoch_data in Gradient_Matrix.items():
                        for layer, layer_data in epoch_data.items():

                            # Calculate the average gradient for all nodes in the layer for the current epoch
                            average_gradient = np.mean([node_data for node_data in layer_data.values()], axis=0)
                            
                            if abs(average_gradient) < 0.02:
                                # print("Gradient stop criteria has been met in layer: ", layer)
                                return Gradient_Matrix, average_energy_error
        return Gradient_Matrix, average_energy_error

        

    def train(self, T, YT, Ne, method, model, learning_rate, show = False, plot = True):
        """
        Train a neural network using the specified training method and parameters.

        This method trains the neural network using the given training method (sequential or batch) over a
        specified number of epochs. It collects and returns the gradients for each layer during training and
        optionally plots them by layer.

        Parameters:
        - T (list): List of input data samples for training.
        - YT (list): List of desired output data corresponding to the training samples.
        - Ne (int): Number of epochs for training.
        - method (str): Training method to use ('Sequential' or 'Batch').
        - learning_rate (float): Learning rate used for weight updates during backpropagation.
        - show (bool): If True, display debugging information during training.
        - plot (bool): If True, plot the gradients by layer after training.

        Returns:
        - dict: A dictionary containing gradients for each layer at each epoch, indexed by the epoch number.
        """
        
        Gradient_Matrix, average_energy_error = self.epoch(T, YT, Ne, method, learning_rate, show)
        if show:
            print('-------------', model, '------------')
        if plot:
            self.plot_by_layers(Gradient_Matrix, model)



        return Gradient_Matrix, average_energy_error
    
    def test(self, T, YT, t, Yt, model, tolerance = 1*10**(-2), show = False, plot = True):
        """
        Test a trained neural network's performance on a dataset.

        This method evaluates the neural network's performance on both the training and testing datasets.
        It calculates the mean squared error for each data point and counts the number of data points with
        errors exceeding the specified tolerance. It also calculates the percentage of data points with
        errors exceeding the tolerance and displays the results if requested.

        Parameters:
        - T (list): List of input data samples for training.
        - YT (list): List of desired output data corresponding to the training samples.
        - t (list): List of input data samples for testing.
        - Yt (list): List of desired output data corresponding to the testing samples.
        - model (str): Name or identifier for the model being tested.
        - tolerance (float): Tolerance level for considering errors acceptable.
        - show (bool): If True, display test results.
        - plot (bool): If True, plot the results.

        Returns:
        - float: Percentage of data points with errors exceeding the tolerance.
        """
        cont = 0
        Errors = []
        Yt_out = []
        # Evaluate the testing dataset
        for x,y in zip(t,Yt):
            out,_,_ = self.forward(x)
            error = np.sum((y - out)**2)
            if abs(error) > tolerance:
                cont+=1
            Errors.append(error)
            Yt_out.append(out)

        # Evaluate the training dataset
        for x,y in zip(T,YT):
            out,_,_ = self.forward(x)
            error = np.sum((y - out)**2)
            if abs(error) > tolerance:
                cont+=1
            Yt_out.append(out)

        average_energy_error = np.mean([(x**2)/2 for x in Errors])
        # Calculate the percentage of data points with errors exceeding the tolerance
        delta = (cont/(len(t)+len(T))*100)
        if show:
            print('---------- Testing ----------')
            print("Mean error: ", np.mean(Errors))
            print("Delta: ", delta , "%")
            print("cont", cont)
        
        return delta, average_energy_error
    
    def iteration(self, X, Y, Ne, method, model, tolerance = 1*10**(-2), max_iter = 50, learning_rate = 1, show = False, plot = True):
        Average_Energies_Training = []
        Average_Energies_Testing = []
        T, YT, t, Yt, V, YV, _, _, _ = self.sampling(X, Y, plot = False)
        for i in range(max_iter):
            
            Gradient_Matrix, average_energy_error = self.train(T, YT, Ne, method, model, learning_rate, plot = False)
            Average_Energies_Training.append(average_energy_error)
            delta, average_energy_error = self.test(T, YT, t, Yt, model, show= False)
            Average_Energies_Testing.append(average_energy_error)
            if delta < tolerance:
                print("Aquí paramos")
                return delta

            Tt = np.vstack((T,t)); YTt = np.vstack((YT,Yt));
            T, YT, t, Yt, _, _, _, _, _ = self.sampling(Tt, YTt, split = [0.75,0.25,0],plot = False)

        if plot:
            self.plot_energy_errors(Average_Energies_Training, Average_Energies_Testing, "Average Energy Errors")
            self.plot_by_layers(Gradient_Matrix,"Gradients")
        return delta, V, YV
    
    def validation(self, V, YV, model, tolerance = 1*10**(-2), show = False, plot = True):
        """
        Validate a trained neural network's performance on a validation dataset.

        This method evaluates the neural network's performance on a validation dataset and calculates the mean
        squared error for each data point. It also counts the number of data points with errors exceeding the
        specified tolerance and calculates the percentage of such points. Additionally, it can plot the errors
        and the comparison between the predicted and actual output values if requested.

        Parameters:
        - V (list): List of input data samples for validation.
        - YV (list): List of desired output data corresponding to the validation samples.
        - model (str): Name or identifier for the model being validated.
        - tolerance (float): Tolerance level for considering errors acceptable.
        - show (bool): If True, display validation results.
        - plot (bool): If True, plot the results.

        Returns:
        - float: Percentage of data points with errors exceeding the tolerance.
        """

        cont = 0
        Errors = []
        YV_out = []

        # Evaluate the validation dataset
        for x,y in zip(V,YV):
            out,_,_ = self.forward(x)
            error = np.sum((y - out)**2)
            if abs(error) > tolerance:
                cont+=1
            Errors.append(error)
            YV_out.append(out)

        # Calculate the percentage of data points with errors exceeding the tolerance
        epsilon = (cont/(len(V))*100)

        if show:
            print('---------- Validation ----------')
            print("Mean error: ", np.mean(Errors))
            print("Epsilon: ", epsilon, "%")
            print("cont", cont)

        if plot:
            X = np.arange(len(Errors))
            MeanE = [np.mean(Errors)]*len(Errors)
            plt.plot(X, Errors, linestyle = '-', color='darkgrey', label = 'Error per point')
            plt.plot(X, MeanE, linestyle = '--', color='black', label = 'Mean error')
            plt.xlabel('Point')
            plt.ylabel('Cuadratic Error')
            plt.legend()
            plt.title('Cuadratic Error for Validation Set')
            plt.savefig('Cuadratic Error for Validation Set')
            plt.show()

            self.plot_curve(YV, YV_out, 'Predicted vs Estimated Validated Set')
        
        return epsilon

    def sampling(self, X, Y, split = [0.6,0.2,0.2], plot=False):
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
        lenT, lent, lenV = round(len(X)*split[0]), round(len(X)*split[1]), round(len(X)*split[2])

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


            fig, axs = plt.subplots(3, 1, figsize=(8, 10))

            # Plot for the Training Set
            ax = axs[0]
            ax.plot(Y, X[:, 0], linestyle='-', color='darkgrey')
            ax.scatter(YT, T[:, 0], marker='*', color='black')
            ax.plot(Y, X[:, 1], linestyle='-', color='darkgrey')
            ax.scatter(YT, T[:, 1], marker='v', color='black')
            ax.set_title('Training set')
            ax.set_xlabel('t')
            ax.legend(['COD1', 'Sampled COD1', 'Xv1', 'Sampled Xv1'])

            # Plot for the Testing Set
            ax = axs[1]
            ax.plot(Y, X[:, 0], linestyle='-', color='darkgrey')
            ax.scatter(Yt, t[:, 0], marker='*', color='black')
            ax.plot(Y, X[:, 1], linestyle='-', color='darkgrey')
            ax.scatter(Yt, t[:, 1], marker='v', color='black')
            ax.set_title('Testing set')
            ax.set_xlabel('t')
            ax.legend(['COD1', 'Sampled COD1', 'Xv1', 'Sampled Xv1'])

            # Plot for the Validation Set
            ax = axs[2]
            ax.plot(Y, X[:, 0], linestyle='-', color='darkgrey')
            ax.scatter(YV, V[:, 0], marker='*', color='black')
            ax.plot(Y, X[:, 1], linestyle='-', color='darkgrey')
            ax.scatter(YV, V[:, 1], marker='v', color='black')
            ax.set_title('Validation set')
            ax.set_xlabel('t')
            ax.legend(['COD1', 'Sampled COD1', 'Xv1', 'Sampled Xv1'])

            # Set the title above the subplots
            plt.suptitle("Data Partitioning and Sampling", fontsize=16)
            plt.subplots_adjust(top=0.8)  # Adjust the position of the suptitle

            # Save and display the plot
            plt.tight_layout()
            plt.savefig('Data_Partitioning_and_Sampling.png')
            plt.show()

        return T, YT, t, Yt, V, YV, indT, indt, indV

    def plot_by_neuron(self, Gradient_Matrix, save_path):
        # Initialize the new dictionary
        Neuron_Gradients = {}

        # Iterate through epochs, layers, and nodes
        for epoch, epoch_data in Gradient_Matrix.items():
            for layer, layer_data in epoch_data.items():
                for node, node_data in layer_data.items():
                    # Create a unique key for each combination of layer and node
                    combined_key = f"{layer}, {node}"
                    
                    # If the key is not in Neuron_Gradients, initialize it with an empty list
                    if combined_key not in Neuron_Gradients:
                        Neuron_Gradients[combined_key] = []
                    
                    # Append the data from the current epoch, layer, and node to the list
                    Neuron_Gradients[combined_key].extend(node_data)

        # Get the unique combinations of layer and node
        neurons = list(Neuron_Gradients.keys())

        # Plotting in separate subplots
        num_neurons = len(neurons)
        num_columns = 2  # Number of columns for subplots
        num_rows = (num_neurons + num_columns - 1) // num_columns  # Calculate number of rows

        fig, axs = plt.subplots(num_rows, num_columns, figsize=(5, 3*num_rows))
        axs = axs.flatten()  # Flatten the 2D axs array to 1D

        for idx, combination in enumerate(neurons):
            values = Neuron_Gradients[combination]
            ax = axs[idx]
            ax.plot(values, linestyle = '--', color='black')
            ax.set_title(combination)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Values')

        # Hide any unused subplots if necessary
        for i in range(num_neurons, num_rows * num_columns):
            axs[i].axis('off')

        plt.tight_layout()

        plt.suptitle(save_path)
        plt.subplots_adjust(top=0.9) 

        # Save the figure to the specified path (replace 'save_path' with your desired file path and format)
        plt.savefig(save_path)

        # Display the plot
        plt.show()

    def plot_by_layer(self, Gradient_Matrix, save_path):
        # Initialize the new dictionary
        Layer_Gradients = {}

        # Iterate through epochs, layers, and nodes
        for epoch, epoch_data in Gradient_Matrix.items():
            for layer, layer_data in epoch_data.items():
                # Create a unique key for each layer
                combined_key = f"{layer}"
                
                # If the key is not in Layer_Gradients, initialize it with an empty list
                if combined_key not in Layer_Gradients:
                    Layer_Gradients[combined_key] = []
                
                # Calculate the average gradient for all nodes in the layer for the current epoch
                average_gradient = np.mean([node_data for node_data in layer_data.values()], axis=0)
                
                # Append the average gradient data to the list
                Layer_Gradients[combined_key].append(average_gradient)

        # Get the unique layers
        layers = list(Layer_Gradients.keys())

        # Plotting in separate subplots
        num_layers = len(layers)
        num_columns = num_layers  # Number of columns for subplots
        num_rows = 1  # Number of rows for subplots (we want them all in the same row)

        fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 5))
        
        # Ensure axs is a list even for a single subplot
        axs = [axs] if num_layers == 1 else axs

        for idx, layer in enumerate(layers):
            values = Layer_Gradients[layer]
            ax = axs[idx]
            ax.plot(values, linestyle = '-', color='black')
            ax.set_title(layer)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Average Gradient')

        plt.tight_layout()

        plt.suptitle(save_path)
        plt.subplots_adjust(top=0.9) 

        # Save the figure to the specified path (replace 'save_path' with your desired file path and format)
        plt.savefig(save_path)

        # Display the plot
        plt.show()

    def plot_by_layers(self, Gradient_Matrix, save_path):
        # Initialize the new dictionary
        Layer_Gradients = {}

        # Iterate through epochs, layers, and nodes
        for epoch, epoch_data in Gradient_Matrix.items():
            for layer, layer_data in epoch_data.items():
                # Create a unique key for each layer
                combined_key = f"{layer}"
                
                # If the key is not in Layer_Gradients, initialize it with an empty list
                if combined_key not in Layer_Gradients:
                    Layer_Gradients[combined_key] = []
                
                # Calculate the average gradient for all nodes in the layer for the current epoch
                average_gradient = np.mean([node_data for node_data in layer_data.values()], axis=0)
                
                # Append the average gradient data to the list
                Layer_Gradients[combined_key].append(average_gradient)

        # Get the unique layers
        layers = list(Layer_Gradients.keys())

        # Plotting in separate subplots
        num_layers = len(layers)
        lines = ['-', '--', '-.', ':', '--']

        for idx, layer in enumerate(layers):
            values = Layer_Gradients[layer]
            line = lines[idx]
            plt.plot(values, linestyle = line, color='black', label = layer)

            plt.xlabel('Epoch')
            plt.ylabel('Average Gradient')

        plt.tight_layout()

        plt.suptitle(save_path)
        plt.subplots_adjust(top=0.9) 
        plt.legend()

        # Save the figure to the specified path (replace 'save_path' with your desired file path and format)
        plt.savefig(save_path)

        # Display the plot
        plt.show()
    def plot_curve(self, Y_desired, Y_out, save_path, label1 = 'Y desired', label2 = 'Y estimated'):
        # Create an array of indices based on the length of Y_out
        X = np.arange(len(Y_out))
        
        # Plot the desired and estimated curves
        plt.figure(figsize=(8, 6))  # Specify the figure size (adjust as needed)
        plt.plot(X, Y_desired, linestyle='-', color='darkgrey', label=label1)
        plt.plot(X, Y_out, linestyle='--', color='black', label= label2)
        
        # Add a title and labels to the plot
        plt.title(save_path)
        plt.xlabel('Point')
        plt.ylabel('Y value')
        
        # Add a legend to distinguish between the two curves
        plt.legend()
        
        # Adjust the subplot layout for better spacing
        plt.tight_layout()
        
        # Save the plot to the specified path (optional)
        plt.savefig(save_path)
        
        # Display the plot (you can remove this line if you only want to save the plot)
        plt.show()

    def plot_energy_errors(self, Y_desired, Y_out, save_path):
        # Create an array of indices based on the length of Y_out
        X = np.arange(len(Y_out))
        
        # Plot the desired and estimated curves
        plt.figure(figsize=(8, 6))  # Specify the figure size (adjust as needed)
        plt.plot(X, Y_desired, linestyle='-', color='darkgrey', label= 'Training')
        plt.plot(X, Y_out, linestyle='--', color='black', label= 'Testing')
        
        # Add a title and labels to the plot
        plt.title(save_path)
        plt.xlabel('Epoch')
        plt.ylabel('Energy Error')
        
        # Add a legend to distinguish between the two curves
        plt.legend()
        
        # Adjust the subplot layout for better spacing
        plt.tight_layout()
        
        # Save the plot to the specified path (optional)
        plt.savefig(save_path)
        
        # Display the plot (you can remove this line if you only want to save the plot)
        plt.show()