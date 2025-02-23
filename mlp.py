import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple


def batch_generator(train_x, train_y, batch_size):
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n, f).
    :param train_y (np.ndarray): Target values of shape (n, q).
    :param batch_size (int): The size of each batch.

    :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
    """
    for i in range(0, len(train_x), batch_size):
        batch_x = train_x[i:i + batch_size]
        batch_y = train_y[i:i + batch_size]
        yield batch_x, batch_y


class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the output of the activation function, evaluated on x

        Input args may differ in the case of softmax

        :param x (np.ndarray): input
        :return: output of the activation function
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function, evaluated on x
        :param x (np.ndarray): input
        :return: activation function's derivative at x
        """
        pass


class Sigmoid(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    def derivative(self, x: np.ndarray) -> np.ndarray:
        sig = self.forward(x)
        return sig * (1 - sig)


class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2


class Relu(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)


class Softmax(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return np.array([np.diagflat(s_i) - np.outer(s_i, s_i) for s_i in s])


class Linear(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x).astype(np.float64)

class SoftPlus(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.log(1 + np.exp(x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
class Mish(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        sp = SoftPlus()
        return x * np.tanh(sp.forward(x))

    #sech^2 (soft plus(x)) * x * sigmoid(x) + forward(x) / x
    def derivative(self, x: np.ndarray) -> np.ndarray:
        sp = SoftPlus()
        sigmoid = Sigmoid()
        sech2 = 1 / np.cosh(sp.forward(x)) ** 2
        return sech2 * x * sigmoid.forward(x) + self.forward(x) / x




class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sum(((y_true - y_pred) ** 2)/2) / float(len(y_true))
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return (y_true - y_pred) / float(len(y_true))


class CrossEntropy(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -y_true / (y_pred + 1e-9)


class Layer:
    def __init__(
            self, 
            fan_in: int, 
            fan_out: int, 
            activation_function: ActivationFunction, 
            verbose :bool = False,  
            is_last :bool = False,
            is_input :bool = False,
            layer_id :int = 1,
            training_dropout :float = 0.0
            ):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        :param verbose: print what happened at the layer.
        :param loss_type: the type of loss (MSE or CE). Needed to computer gradients
        :param is_last: if the layer is the last layer in the network or not
        :param is_input: if the layer is the input layer or not
        :param loss_grads: loss gradients, used for backpropagation if it's the last layer
        :param layer_id: The index of the layer, for debugging purposes
        :param training_dropout: the dropout rate for the layer during training.
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function
        #self.loss_function = loss_function
        self.is_last = is_last
        self.is_input = is_input
        self.verbose = verbose
        self.layer_id = layer_id
        self.dropout = training_dropout
        # this will store the activations (forward prop)
        self.activations = None
        # this will store the delta term (dL_dPhi, backward prop)
        self.delta = None
        self.loss_grads :np.ndarray = None
        # Initialize weights and biaes
        limit = np.sqrt(6 / (fan_in + fan_out))
        self.W :np.ndarray = np.random.uniform(-limit, limit, (fan_in, fan_out))  # Glorot uniform initialization for weights
        self.b :np.ndarray = np.zeros((1, fan_out))  # Initialize biases to zero

        self.cache_W = np.zeros_like(self.W)
        self.cache_b = np.zeros_like(self.b)

    def forward(self, h: np.ndarray):
        """
        Computes the activations for this layer

        :param h: input to layer
        :return: layer activations
        """
        if self.verbose:
            print(f"Layer {self.layer_id} Feed forward. In:  {h.shape}. Weights: {self.W.shape}\n")
        if self.is_input:
            return h
        if self.dropout > 0.0 and not self.is_last:
            self.mask = np.random.binomial(1, 1 - self.dropout, size=h.shape) / (1 - self.dropout)
            
            
            h *= self.mask
        self.z = np.dot(h, self.W) + self.b
        self.activations = self.activation_function.forward(self.z)
        if self.verbose:
            print(f"Feed forward. Out:{self.activations.shape}\n\n")
        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backpropagation to this layer and return the weight and bias gradients

        :param h: input to this layer
        :param delta: delta term from layer above
        :return: (weight gradients, bias gradients)
        """
        if self.verbose:
            print(f"Layer {self.layer_id} Backward start.")
        #computing derivatives
        
        z = self.z
        if self.verbose:
            print(f"Raw output: {z.shape}")
        d_out_wrt_raw = self.activation_function.derivative(z)
        
        d_raw_wrt_w = h
        d_raw_wrt_input = self.W
        if self.verbose and self.loss_grads is not None:
            print(f"d_loss_wrt_out: {self.loss_grads.shape}\n d_out_wrt_raw: {d_out_wrt_raw.shape}")

        #take care of the math issues for a softmax derivative
        if isinstance(self.activation_function, Softmax):
            
            incoming_delta = self.loss_grads if self.is_last else delta
            
            jacobian = self.activation_function.derivative(z)
            
            hadamard = np.einsum('bij,bi->bj', jacobian, incoming_delta)
        else:
            if self.is_last:
                d_loss_wrt_out = self.loss_grads
                hadamard = np.multiply(d_loss_wrt_out, d_out_wrt_raw)
            else:
                hadamard = np.multiply(delta, d_out_wrt_raw)

        
        if self.verbose:
            print(f"Hadamard product: {hadamard.shape}\n d_raw_wrt_input: {d_raw_wrt_input.shape}")
        self.delta = np.dot(hadamard, d_raw_wrt_input.transpose())
        dL_dW = np.dot(d_raw_wrt_w.transpose(), hadamard)
        #print(f"dL_dW: {dL_dW.shape}")

        #print(f"d_raw_wrt_bias: {np.ones_like(h).shape}")
        dL_db = np.sum(hadamard, axis=0, keepdims=True)

        if self.verbose:
            print(f"Layer {self.layer_id} Backward output. Delta: {self.delta.shape}\nweight change: {dL_dW.shape}\n Bias change: {dL_db.shape}")
        return dL_dW, dL_db


class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer], verbose=False):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        :param verbose: verbose output for debugging
        """
        self.verbose = verbose
        layers[0].is_input = True
        layers[-1].is_last = True
        for j, layer in enumerate(layers):
            layers[j].layer_id = j + 1
            layers[j].verbose = self.verbose
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input
        :return: network output
        """
        output = None
        for layer in self.layers:
            if output is None:
                output = layer.forward(x)
            else:
                output = layer.forward(output)


        return output
    
    def forward_with_layer_inputs(self, x: np.ndarray) -> Tuple[np.ndarray]:
        """
        This takes the network input and computes the network output, but also returns the inputs at each layer for use in 
        backpropagation.
        :param x: network input
        :return: network output
        """
        if self.verbose:
            print("Starting forward propagation...")

        output = None
        layer_inputs = [x]
        for layer in self.layers:
            if output is None:
                output = layer.forward(x)
            else:
                output = layer.forward(output)
            layer_inputs.append(output)


        return (output, layer_inputs)

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray, layer_inputs :np.ndarray) -> Tuple[list, list]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param input_data: network's input data
        :param layer_inputs: inputs at each layer
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        if self.verbose:
            print("Starting backpropagation...")
        dl_dw_all = []
        dl_db_all = []
        current_delta = None
        for i, layer in enumerate(reversed(self.layers)):
            if layer.is_input:
                break
            if layer.is_last:
                layer.loss_grads = loss_grad
            dL_dW, dL_db = layer.backward(layer_inputs[-2 - i], current_delta)
            current_delta = layer.delta
            dl_dw_all.insert(0, dL_dW)
            dl_db_all.insert(0, dL_db)

        return dl_dw_all, dl_db_all


    def train(self, 
              train_x: np.ndarray, 
              train_y: np.ndarray, 
              val_x: np.ndarray, 
              val_y: np.ndarray, 
              loss_func: LossFunction, 
              learning_rate: float=1E-3, 
              batch_size: int=16, 
              epochs: int=32,
              decay_rate: float = 0.9, 
              epsilon: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the multilayer perceptron

        :param train_x: full training set input of shape (n x d) n = number of samples, d = number of features
        :param train_y: full training set output of shape (n x q) n = number of samples, q = number of outputs per sample
        :param val_x: full validation set input
        :param val_y: full validation set output
        :param loss_func: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :param decay_rate: RMSProp decay factor (typically around 0.9)
        :param epsilon: small constant to prevent division by zero
        :return:
        """
        training_losses = []
        validation_losses = []
        print(f"train_y: {train_y.shape}")
        for layer in self.layers:
            if not layer.is_input:
                if not hasattr(layer, 'cache_W'):
                    layer.cache_W = np.zeros_like(layer.W)
                if not hasattr(layer, 'cache_b'):
                    layer.cache_b = np.zeros_like(layer.b)

        for i in range(epochs):
            print(f"Starting epoch {i}...")
            current_losses = []
            for batch_x, batch_y in batch_generator(train_x, train_y, batch_size):
                batch_output, layer_inputs = self.forward_with_layer_inputs(batch_x)
                #print(f"Batch output: {batch_output.shape}")
                #print(f"Batch_y: {batch_y.shape}")
                batch_losses = loss_func.loss(batch_y, batch_output)
                loss_gradients = loss_func.derivative(batch_y, batch_output)
                if isinstance(self.layers[-1].activation_function, Softmax):
                    if self.verbose:
                        print("Last layer is softmax. Doing the easy calculation.")
                    loss_gradients = -(batch_output - batch_y)
                weight_gradients, bias_gradients = self.backward(loss_gradients, batch_x, layer_inputs)
                
                #perform weight and bias updates
                for j, layer in enumerate(reversed(self.layers)):
                    if layer.is_input:
                        break
                    if layer.W.shape == weight_gradients[-1 - j].shape:
                        layer.W += learning_rate * weight_gradients[-1 - j]
                    else:
                        print(f"Shape mismatch in layer {len(self.layers)-j}: layer.W.shape = {layer.W.shape}, weight_gradients[{len(self.layers)-j}].shape = {weight_gradients[j].shape}")
                        assert 1 == 0
                    
                    if layer.b.shape == bias_gradients[-1 - j].shape:
                        layer.b += learning_rate * bias_gradients[-1 - j] 
                    else:
                        print(f"Shape mismatch in layer {len(self.layers)-j }: layer.b.shape = {layer.b.shape}, bias_gradients[{len(self.layers)-j}].shape = {bias_gradients[j].shape}")
                        assert 1 == 0
                    
                current_losses.append(batch_losses)
        
            avg_train_loss = np.mean(current_losses)
            print(f"Training loss for epoch {i} : {avg_train_loss}")
            training_losses.append(avg_train_loss)
            
                
                


            #compute avg validation loss
            val_losses = []
            for v in range(len(val_x)):
                val_loss = loss_func.loss(self.forward(val_x[v]), val_y[v])
                val_losses.append(val_loss)
            avg_val_loss = np.average(val_losses)
            print(f"Validation loss for epoch {i}: {avg_val_loss}")
            validation_losses.append(avg_val_loss)

        return training_losses, validation_losses

