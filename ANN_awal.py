import numpy as np

def inisialisasi_parameter(input_size, hidden_size, output_size):
    np.random.seed(42)
    parameter = {
        "W1": np.random.randn(hidden_size, input_size) * 0.01,
        "b1": np.zeros((hidden_size, 1)),
        "W2": np.random.randn(output_size, hidden_size) * 0.01,
        "b2": np.zeros((output_size, 1))
    }
    return parameter

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(int)

def forward_propagation(X, parameter):
    W1, b1, W2, b2 = parameter["W1"], parameter["b1"], parameter["W2"], parameter["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

def komputasi_biaya(Y, A2):
    m = Y.shape[1]
    biaya = -np.sum(Y * np.log(A2) + (1-Y) * np.log(1-A2)) / m
    return np.squeeze(biaya)

def backward_propagation(X, Y, parameter, cache):
    m = X.shape[1]
    W2 = parameter["W2"]

    dZ2 = cache["A2"] - Y
    dW2 = np.dot(dZ2, cache["A1"].T) /m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = np.dot(W2.T, dZ2) * relu_derivative(cache["Z1"])
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1, "db1":db1, "dW2":dW2, "db2":db2}
    return grads

def parameter_terbaru(parameter, grads, learning_rate):
    for key in parameter.keys():
        parameter[key] -= learning_rate * grads["d" + key]
    return parameter

def train_neural_network(X, Y, input_size, hidden_size, output_size, epochs=1000, learning_rate=0.01):
    parameter = inisialisasi_parameter(input_size, hidden_size, output_size)

    for i in range(epochs):
        A2, cache = forward_propagation(X, parameter)
        biaya = komputasi_biaya(Y, A2)
        grads = backward_propagation(X, Y, parameter, cache)
        parameter = parameter_terbaru(parameter, grads, learning_rate)

        if i % 100 == 0 :
            print(f"Epoch {i}: Cost = {biaya}")
    return parameter


def predict (X, parameter):
    A2, _ = forward_propagation(X, parameter)
    return (A2 > 0.5).astype(int)

X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
Y = np.array([[0, 0, 0, 1]])

trained_parameters = train_neural_network(
    X, Y, input_size=2, hidden_size=4, output_size=1, epochs=10000, learning_rate=0.1
)

prediksi = predict(X, trained_parameters)
print("Predictions: ", prediksi)