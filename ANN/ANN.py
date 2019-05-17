import numpy as np
def sigmoid(x):
    return (1/(1+np.exp(-x)))

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def sigmoid_dash(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh_dash(x):
    return 1-np.tanh(x)*np.tanh(x)

Wih = np.random.rand(3,3)-0.5
Who = np.random.rand(3)-0.5

inputs = np.array([[1,-1,1],[1,1,-1],[1,-1,-1],[1,1,1]])
targets = np.array([1,1,-1,-1])

def forward_pass(inputs, Wih, Who):
    h = Wih @ inputs.T
    v = sigmoid(h)
    y = -1+2*sigmoid(Who.T @ v)
    return y, v

def gradient_descent_step(Wih,Who,inputs,targets,learning_rate=0.01):
    y,v = forward_pass(inputs,Wih,Who)
    for i in range(len(inputs)):
        Who -= -learning_rate*4*(targets[i] - y[i])*sigmoid_dash(y[i])*v.T[i]
        Wih -= -learning_rate*4*(targets[i] - y[i])*Who*(sigmoid_dash(y[i])*sigmoid_dash(v.T[i]))@inputs[i]
    return Wih, Who

def train(Who, Wih, inputs, targets, n_iter):
    for i in range(n_iter):
        Wih, Who = gradient_descent_step(Wih,Who,inputs,targets)

for i in range(len(inputs)):
    train(Who, Wih, inputs,targets,n_iter=600)


for i in range(len(inputs)):
    y, _ = forward_pass(inputs[i],Wih,Who)
    print(y)
