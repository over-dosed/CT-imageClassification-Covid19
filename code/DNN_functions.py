import numpy as np

def initialize_parameters(n_x, n_h, n_y): # for 2 layers(relu -> sigmoid)
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    
    W1 = np.random.randn(n_h , n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)* 0.01
    b2 = np.zeros((n_y,1))
    
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters    

def initialize_parameters_deep(layer_dims): # for a deep network
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def linear_forward(A, W, b): 

    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def sigmoid(Z):
    a = 1/(1 + np.exp(-Z))
    cache = Z

    return a,cache

def relu(Z):
    a = np.maximum(0,Z)
    cache = Z

    return a,cache

def softmax(Z):
    """ softmax function """
    a = Z
    a -= np.max(Z, axis = 0, keepdims = True) #为了稳定计算softmax概率， 减掉最大的那个元素
    
    a = np.exp(a) / np.sum(np.exp(a), axis = 0, keepdims = True)
    cache = Z
    return a,cache
        
def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    if activation == "sigmoid":
        
        Z , linear_cache = linear_forward(A_prev,W,b)
        A , activation_cache = sigmoid(Z)
        
    elif activation == "relu":
        
        Z , linear_cache = linear_forward(A_prev,W,b)
        A , activation_cache = relu(Z)
    
    elif activation == "softmax":
        
        Z , linear_cache = linear_forward(A_prev,W,b)
        A , activation_cache = softmax(Z)
    
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2                 

    for l in range(1, L):
        A_prev = A 
        
        A, cache = linear_activation_forward(A_prev, parameters["W"+ str(l)], parameters["b"+ str(l)], "relu")
        caches.append(cache) 
    
    AL, cache = linear_activation_forward(A, parameters["W"+ str(L)], parameters["b"+ str(L)], "softmax")
    caches.append(cache) 

    return AL, caches

def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]
    
    cost = (-1 / m) * ( np.dot(Y, np.log(AL).T) + np.dot(1-Y, np.log(1 - AL).T)) 
    cost = np.squeeze(cost)   
    return cost

def computer_cost_softmax(AL,Y):

    m = Y.shape[1]
    cost = np.sum(AL * Y,axis=0)
    cost = np.sum(cost) * (-1 / m)
    cost = np.squeeze(cost)   
    return cost

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot( dZ , A_prev.T) / m
    db = np.sum(dZ,axis=1, keepdims=True) / m
    dA_prev = np.dot(cache[1].T ,dZ)

    return dA_prev, dW, db

def sigmoid_backward(dA , activation_cache):
    a = 1/(1 + np.exp(-activation_cache))
    dZ = dA * (a* (1-a))

    return dZ

def relu_backward(dA , activation_cache):

    dZ = np.where(activation_cache > 0, 1, 0)
    dZ = dA * dZ

    return dZ

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":

        dZ = relu_backward(dA, activation_cache)
        dA_prev,dW,db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":

        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev,dW,db = linear_backward(dZ, linear_cache)
        
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    #dAL = np.zeros((Y.shape)) - (1 / np.max((AL*Y),axis=0))

    current_cache = caches[L-1]
    # dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")
    
    dZ = AL - Y # softmax 对交叉熵损失函数的求导得出
    dA_prev_temp, dW_temp, db_temp = linear_backward(dZ, current_cache[0])

    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp

    return grads

def update_parameters(params, grads, learning_rate):
    
    parameters = params.copy()
    L = len(parameters) // 2 
    for l in range(L):

        parameters["W" + str(l+1)] = params["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = params["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

def predict(AL,Y):
    AL -= np.max(AL, axis = 0, keepdims = True) #每一列减去最大值
    r = AL + Y #相机，计算出现的1的数量即为命中数
    result =(r == 1).sum()
    result /= Y.shape[1]
    return result
