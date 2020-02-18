import numpy as np

X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
y=np.array([[1],[1],[0]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

epochs = 50000
alpha = 0.01
input_neurons = 4
output_neuros = 1
hidden_neurons = 3

wh = np.random.uniform(size=(input_neurons,hidden_neurons))
bh = np.random.uniform(size=(1,hidden_neurons))
wout = np.random.uniform(size=(hidden_neurons,output_neuros))
bout = np.random.uniform(size=(1,output_neuros))

for i in range(epochs):
#    Feedforward
    z1 = np.dot(X,wh) + bh
    hidden_layer = sigmoid(z1)
    
    z2 = np.dot(hidden_layer,wout)
    output_layer = sigmoid(z2)

#    Backpropagation    
    E = y - output_layer
    
    slope_output_layer = sigmoid_derivative(output_layer)
    slope_hidden_layer = sigmoid_derivative(hidden_layer)
    
    d_output = E * slope_output_layer
    error_hidden_layer = d_output.dot(wout.T)
    d_hidden_layer = error_hidden_layer * slope_hidden_layer
    
    wout += hidden_layer.T.dot(d_output) * alpha
    bout += np.sum(d_output, axis=0, keepdims=True) * alpha
    wh += X.T.dot(d_hidden_layer) * alpha
    bh += np.sum(d_hidden_layer) * alpha

print(output_layer)