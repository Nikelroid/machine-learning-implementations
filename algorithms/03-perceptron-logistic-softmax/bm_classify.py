import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):

    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0
 
    if loss == "perceptron":
        y_prime = 2 * y - 1
        for _ in range(max_iterations):
            s = X @ w + b
            errors = (y_prime * s <= 0).astype(float)
            grad_w = - (y_prime * errors) @ X / N
            grad_b = - np.sum(y_prime * errors) / N
            w -= step_size * grad_w
            b -= step_size * grad_b    


    elif loss == "logistic":
        for _ in range(max_iterations):
            linear_combination = np.dot(X , w)+ b
            y_pred = sigmoid(linear_combination)
            derative_w = 1/N * (np.dot(X.T , (y_pred - y)))
            derative_b = 1/N * np.sum(y_pred - y)
            w -= step_size * derative_w
            b -= step_size * derative_b     


    else:
        raise "Undefined loss function."

    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    value = 1 / (1 + np.exp(-z))  
    return value

def binary_predict(X, w, b):
    probs = sigmoid(np.dot(X, w) + b)
    N, D = X.shape
    assert probs.shape == (N,) 
    return (probs >= 0.5).astype(int)

def multiclass_train(X, y, C, w0=None, b0=None, gd_type="sgd", step_size=0.5, max_iterations=1000):
    N, D = X.shape
    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0
    
    np.random.seed(42)
    
    if gd_type == "sgd":
        for _ in range(max_iterations):
            n = np.random.choice(N)
            
            scores = np.dot(w, X[n]) + b
            exp_scores = np.exp(scores - np.max(scores))
            softmax_probs = exp_scores / np.sum(exp_scores)
            
            y_one_hot = np.zeros(C)
            y_one_hot[y[n]] = 1
            
            grad_w = np.outer((softmax_probs - y_one_hot), X[n])
            grad_b = softmax_probs - y_one_hot
            
            w -= step_size * grad_w
            b -= step_size * grad_b
            
    elif gd_type == "gd":
        for _ in range(max_iterations):
            scores = np.dot(X, w.T) + b
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            softmax_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            
            y_one_hot = np.zeros((N, C))
            y_one_hot[np.arange(N), y] = 1
            
            grad_w = np.dot((softmax_probs - y_one_hot).T, X) / N
            grad_b = np.sum(softmax_probs - y_one_hot, axis=0) / N
            
            w -= step_size * grad_w
            b -= step_size * grad_b
            
    else:
        raise "Undefined algorithm."
        
    assert w.shape == (C, D)
    assert b.shape == (C,)
    
    return w, b

def multiclass_predict(X, w, b):
    N, D = X.shape
    scores = np.dot(X, w.T) + b
    preds = np.argmax(scores, axis=1)
    
    assert preds.shape == (N,)
    return preds