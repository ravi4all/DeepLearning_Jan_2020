import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

np.random.seed(0)
X,y = make_moons(400, noise=0.1)

logistic = LogisticRegression()
logistic.fit(X,y)

def plot_decision_boundary(pred_func):
    x1 = np.arange(min(X[:,0]) - 1, max(X[:,0]) + 1, 0.01)
    x2 = np.arange(min(X[:,1]) - 1, max(X[:,1]) + 1, 0.01)
    xx,yy = np.meshgrid(x1,x2)
    z = pred_func(np.c_[xx.flatten(), yy.flatten()])
    z = z.reshape(xx.shape)
    plt.contourf(xx,yy,z,cmap=plt.cm.Spectral)
    plt.scatter(X[:,0],X[:,1],c=y,cmap=ListedColormap(('white','red')))

svm = SVC(kernel='rbf',gamma='auto')
svm.fit(X,y)

# plot_decision_boundary( lambda x : svm.predict(x))

num_examples = len(X)
input_neuron = 2
output_neuron = 2
alpha = 0.01

def calculate_loss(model):
    w1,b1,w2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = np.dot(X,w1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1,w2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
#     loss
    correct_logprob = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprob)
    return data_loss

def predict(model,x):
    w1,b1,w2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = np.dot(X,w1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1,w2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs,axis=1)

def build_model(hidden_layer,epochs,print_loss=False):
    np.random.seed(0)
    W1 = np.random.randn(input_neuron, hidden_layer)
    b1 = np.zeros((1,hidden_layer))
    W2 = np.random.randn(hidden_layer,output_neuron)
    b2 = np.zeros((1,output_neuron))
    model = {}
    
    for i in range(epochs):
        # forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1,keepdims=True)
        
        # Backpropagation
        delta3 = probs
        delta3[range(num_examples),y] -= 1
        delta_W2 = (a1.T).dot(delta3)
        delta_b2 = np.sum(delta3, axis = 0, keepdims=True)
        
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1,2))
        delta_W1 = np.dot(X.T, delta2)
        delta_b1 = np.sum(delta2, axis = 0)
        
        # Adding regularization
#         delta_W2 += reg_lambda * W2
#         delta_W1 += reg_lambda * W1
        
        # Gradient Descent parameter update
        W1 += -alpha * delta_W1
        b1 += -alpha * delta_b1
        W2 += -alpha * delta_W2
        b2 += -alpha * delta_b2
        
        model = {'W1':W1,'b1':b1,'W2':W2,'b2':b2}
        
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i : %f"%(i, calculate_loss(model)))
    
    return model

# Network with hidden layer of size 3
model = build_model(3,20000, print_loss=True)

plot_decision_boundary(lambda x : predict(model,x))




