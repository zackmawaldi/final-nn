import numpy as np

from nn.nn import NeuralNetwork
from nn import preprocess


# init nn to test on

nn_arch = [{'input_dim': 2, 'output_dim': 4, 'activation': 'relu'},
           {'input_dim': 4, 'output_dim': 1, 'activation': 'sigmoid'}]


nn = NeuralNetwork(nn_arch=nn_arch, lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='binary_cross_entropy')

def test_single_forward():
    W_curr = np.array([[0.2, -0.5], [0.1, 0.7]])
    b_curr = np.array([[0.2], [-0.1]])
    A_prev = np.array([[-0.5], [0.7]])
    

    # test relu
    A_curr, Z_curr = nn._single_forward(W_curr, b_curr, A_prev, "relu")

    # compare against pre-calculated values
    assert np.allclose( A_curr , np.array([[0], [0.34]]) )
    assert np.allclose( Z_curr , np.array([[-0.25], [0.34]]) )

    
    
    # test sigmoid
    A_curr, Z_curr = nn._single_forward(W_curr, b_curr, A_prev, "sigmoid")

    # compare against pre-calculated values
    assert np.allclose( A_curr , np.array([[0.437823], [0.584190]]) )
    assert np.allclose( Z_curr , np.array([[-0.25], [0.34]]) )



def test_forward():
    X = np.array([[0.5], [-0.5]])

    output, _ = nn.forward(X)

    # compare against pre-calculated value
    assert round( output[0][0], 3 ) == 0.474


def test_single_backprop():
    W_curr = np.array([[0.2, -0.5], [0.1, 0.7]])
    b_curr = np.array([[0.2], [-0.1]])
    A_prev = np.array([[-0.5], [0.7]])

    Z_curr = np.dot(W_curr, A_prev) + b_curr

    dA_curr = np.array([[1.1], [0.9]])

    dA_prev, dW_curr, db_curr = nn._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, "relu")


    # compare against pre-calculated values
    assert np.allclose( dA_prev , np.array([[0.09], [0.63]]) )
    assert np.allclose( dW_curr , np.array([[0,0], [-0.225,0.315]]) )
    assert np.allclose( db_curr , np.array([[0], [0.45]]) )


def test_predict():
    X = np.array([[0.5], [-0.5]])
    output = nn.predict(X)

    assert round( output[0][0], 3 ) == 0.474


def test_binary_cross_entropy():
    y = np.array([[1], [0]])
    y_hat = np.array([[0.8], [0.2]])

    loss = nn._binary_cross_entropy(y, y_hat)

    assert round( loss, 3 ) == 0.223


def test_binary_cross_entropy_backprop():
    y = np.array([[1], [0]])
    y_hat = np.array([[0.8], [0.2]])

    dA = nn._binary_cross_entropy_backprop(y, y_hat)

    assert np.allclose( dA , np.array([[-0.625], [ 0.625]]) )


def test_mean_squared_error():
    y = np.array([[1], [0]])
    y_hat = np.array([[0.8], [0.2]])
    
    loss = nn._mean_squared_error(y, y_hat)

    assert round( loss, 3 ) == 0.02


def test_mean_squared_error_backprop():
    y = np.array([[1], [0]])
    y_hat = np.array([[0.8], [0.2]])

    dA = nn._mean_squared_error_backprop(y, y_hat)

    assert np.allclose( dA , np.array([[-0.2], [ 0.2]]) )


def test_sample_seqs():
    
    # this example should lead to only duplicating C
    seqs   = ['A', 'B', 'C']
    labels = [1, 1, 0]

    upsampled_seqs, upsampled_labels =  preprocess.sample_seqs(seqs,labels )

    # reorder both seqs and label to undo the random shuffeling in the function
    combined = sorted(zip(upsampled_seqs, upsampled_labels), key=lambda x: x[0])

    upsampled_seqs, upsampled_labels = zip(*combined)

    assert upsampled_seqs   == ('A', 'B', 'C', 'C')
    assert upsampled_labels == (1, 1, 0, 0)


def test_one_hot_encode_seqs():
    
    str_seq = ['AGA']
    one_hot = preprocess.one_hot_encode_seqs( str_seq )

    expected = np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0])

    assert np.array_equal( one_hot , expected )