# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        # glorified mx + b
        Z_curr = np.dot(W_curr, A_prev) + b_curr

        if activation == "relu":
            A_curr = self._relu(Z_curr)

        elif activation == "sigmoid":
            A_curr = self._sigmoid(Z_curr)
            
        else:
            raise NotImplementedError(f'Activation function "{activation}" not supported.')
        
        return A_curr, Z_curr
    

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """

        cache = {"A0": X}
        A_curr = X

        for idx, layer in enumerate(self.arch):
            idx += 1

            W_curr = self._param_dict["W" + str(idx)]
            b_curr = self._param_dict["b" + str(idx)]

            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_curr, layer["activation"])

            cache["A" + str(idx)] = A_curr
            cache["Z" + str(idx)] = Z_curr
        
        return A_curr, cache

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """

        # get dL/dZ
        if activation_curr == "relu":
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)

        elif activation_curr == "sigmoid":
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)

        else:
            raise NotImplementedError(f'Activation function "{activation_curr}" not supported.')

        # 1/m to average across the batch
        m = A_prev.shape[0]

        # dZ/dW == A_prev, thus dL/dW == dL/dZ * dZ/dW == np.dot(dZ_curr, A_prev.T) --> T for proper dimentions alignment
        dW_curr = 1/m * np.dot(dZ_curr, A_prev.T)

        # dZ/db == 1, thus dL/db == dL/dZ * 1 == sum(dZ_curr) axis=1 for row wise
        db_curr = 1/m * np.sum(dZ_curr, axis=1, keepdims=True)

        # dZ/dA == W_curr, thus dL/dA == dL/dZ * dZ/dW == np.dot(dZ_curr, W_curr.T) --> T for proper dimentions alignment
        # no batch normal to keep each example's dA_prev uniqueness
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr
    

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """

        grad_dict = {}

        # get intial dA_prev
        if self._loss_func == "binary_cross_entropy":
            dA_prev = self._binary_cross_entropy_backprop(y, y_hat)

        elif self._loss_func == "mean_squared_error":
            dA_prev = self._mean_squared_error_backprop(y, y_hat)

        else:
            raise NotImplementedError(f'Loss function "{self._loss_func}" not supported.')

        # walk backwards
        for idx_prev in range(len(self.arch) - 1, -1, -1):
            idx_curr = idx_prev + 1
            activation_curr = self.arch[idx_prev]['activation']

            dA_curr = dA_prev

            A_prev = cache['A' + str(idx_prev)]
            Z_curr = cache['Z' + str(idx_curr)]
            W_curr = self._param_dict['W' + str(idx_curr)]
            b_curr = self._param_dict['b' + str(idx_curr)]

            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr)

            grad_dict['dW' + str(idx_curr)] = dW_curr
            grad_dict['db' + str(idx_curr)] = db_curr

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for idx, layer in enumerate(self.arch):
            idx += 1
            self._param_dict['W' + str(idx)] -= self._lr * grad_dict['dW' + str(idx)]
            self._param_dict['b' + str(idx)] -= self._lr * grad_dict['db' + str(idx)]



    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """

        train_epoch_loss_list = []
        val_epoch_loss_list = []

        if self._loss_func == "binary_cross_entropy":
            loss_function = self._binary_cross_entropy

        elif self._loss_func == "mean_squared_error":
            loss_function = self._mean_squared_error

        else:
            raise NotImplementedError(f'Loss function "{self._loss_func}" not supported.')
        

        for epoch in range(self._epochs):
            # forward pass
            y_hat, cache = self.forward(X_train)
            loss_train = loss_function(y_train, y_hat)
            train_epoch_loss_list.append(loss_train)

            # backprop
            grad_dict = self.backprop(y_train, y_hat, cache)
            self._update_params(grad_dict)

            # validation
            y_val_hat, _ = self.forward(X_val)
            loss_val = loss_function(y_val, y_val_hat)
            val_epoch_loss_list.append(loss_val)

            print(f"Epoch {epoch + 1}  \tTraining Loss: {loss_train}\tValidation Loss: {loss_val}")

        return train_epoch_loss_list, val_epoch_loss_list


    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """

        y_hat, cache = self.forward(X)

        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        
        ''' Adapted from HW 7 '''

        return 1 / (1 + np.exp(-Z))

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        
        ''' Adapted from HW 7 '''
        
        # web.stanford.edu/~jurafsky/slp3/5.pdf
        # pg 22
        s = np.array(self._sigmoid(Z))
        dA = np.array(dA)

        # last partial * new partial
        return dA * ( s * (1 - s) )

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return Z * (Z > 0)

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        # last partial * new partial
        return dA * ( 1 * (Z > 0) )

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        
        ''' Adapted from HW 7 '''

        m = y.shape[0]

        loss = -1/m * np.sum( y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

        return loss


    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """

        ''' Adapted from HW 7 '''

        # derivation of the gradient, which I will not
        # pretend to understand, is on page 16 and 22 of the following:
        # web.stanford.edu/~jurafsky/slp3/5.pdf

        m = y.shape[0]

        gradient = -(1 / m) * (y / y_hat - (1 - y) / (1 - y_hat))

        return gradient

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """

        m = y.shape[0]
        
        return 1/m * np.mean(np.square(y - y_hat))

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        
        m = y.shape[0]

        return 1/m * 2 * (y_hat - y)
