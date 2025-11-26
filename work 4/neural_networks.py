from utils import softmax_cross_entropy, add_momentum, data_loader_mnist, predict_label, DataSplit
import sys
import os
import argparse
import numpy as np
import json

class linear_layer:

    def __init__(self, input_D, output_D):

        self.params = dict()
        self.gradient = dict()

        self.params['W'] = np.random.normal(0, 0.1, (input_D, output_D))
        self.params['b'] = np.random.normal(0, 0.1, (1, output_D))
        self.gradient['W'] = np.zeros((input_D, output_D))
        self.gradient['b'] = np.zeros((1, output_D))

    def forward(self, X):

        forward_output = self.params['b'] + np.dot(X, self.params['W'])
        return forward_output

    def backward(self, X, grad):

        self.gradient['W'] = np.dot(X.T, grad) 
        self.gradient['b'] = np.sum(grad, axis=0, keepdims=True) 
        backward_output = np.dot(grad, self.params['W'].T) 
        return backward_output



class relu:


    def __init__(self):
        self.mask = None

    def forward(self, X):

        forward_output = np.where(X > 0, X, 0)
        self.mask = (X > 0).astype(float)
        return forward_output

    def backward(self, X, grad):
        backward_output = grad * self.mask
        return backward_output

class tanh:

    def forward(self, X):
        forward_output = np.tanh(X)
        self.mask = 1 - forward_output ** 2
        return forward_output

    def backward(self, X, grad):
        backward_output = grad * self.mask
        return backward_output


# 4. Dropout
class dropout:

    def __init__(self, r):
        self.r = r
        self.mask = None

    def forward(self, X, is_train):

        if is_train:
            self.mask = (np.random.uniform(0.0, 1.0, X.shape) >= self.r).astype(float) * (1.0 / (1.0 - self.r))
        else:
            self.mask = np.ones(X.shape)
        forward_output = np.multiply(X, self.mask)
        return forward_output

    def backward(self, X, grad):
        
        backward_output = np.multiply(grad, self.mask)
        return backward_output



def miniBatchGradientDescent(model, momentum, _alpha, _learning_rate):
    for module_name, module in model.items():
        if hasattr(module, 'params'):
            for key, _ in module.params.items():
                g = module.gradient[key]

                if _alpha <= 0.0:
                    module.params[key] = module.params[key] -  _learning_rate * g
                else:
                    momentum[module_name + '_' + key] = _alpha * momentum[module_name + '_' + key] - _learning_rate * g
                    module.params[key] = module.params[key] + momentum[module_name + '_' + key]

    return model



def main(main_params):

    np.random.seed(int(main_params['random_seed']))

    Xtrain, Ytrain, Xval, Yval , _, _ = data_loader_mnist(dataset = main_params['input_file'])
    N_train, d = Xtrain.shape
    N_val, _ = Xval.shape

    unique, counts = np.unique(Ytrain, return_counts=True)
    counts = dict(zip(unique, counts)).values()

    trainSet = DataSplit(Xtrain, Ytrain)
    valSet = DataSplit(Xval, Yval)

    model = dict()
    num_L1 = 1000
    num_L2 = 10

    num_epoch = int(main_params['num_epoch'])
    minibatch_size = int(main_params['minibatch_size'])

    _learning_rate = float(main_params['learning_rate'])
    _step = 10
    _alpha = float(main_params['alpha'])
    _dropout_rate = float(main_params['dropout_rate'])
    _activation = main_params['activation']


    if _activation == 'relu':
        act = relu
    else:
        act = tanh

    model['L1'] = linear_layer(input_D = d, output_D = num_L1)
    model['nonlinear1'] = act()
    model['drop1'] = dropout(r = _dropout_rate)
    model['L2'] = linear_layer(input_D = num_L1, output_D = num_L2)
    model['loss'] = softmax_cross_entropy()

    if _alpha > 0.0:
        momentum = add_momentum(model)
    else:
        momentum = None

    train_acc_record = []
    val_acc_record = []

    train_loss_record = []
    val_loss_record = []

    for t in range(num_epoch):
        print('At epoch ' + str(t + 1))
        if (t % _step == 0) and (t != 0):
            _learning_rate = _learning_rate * 0.1

        idx_order = np.random.permutation(N_train)

        train_acc = 0.0
        train_loss = 0.0
        train_count = 0

        val_acc = 0.0
        val_count = 0
        val_loss = 0.0

        for i in range(int(np.floor(N_train / minibatch_size))):

            x, y = trainSet.get_example(idx_order[i * minibatch_size : (i + 1) * minibatch_size])

            a1 = model['L1'].forward(x)
            h1 = model['nonlinear1'].forward(a1)
            d1 = model['drop1'].forward(h1, is_train = True)
            a2 = model['L2'].forward(d1)
            loss = model['loss'].forward(a2, y)

            grad_a2 = model['loss'].backward(a2, y)
            grad_d1 = model['L2'].backward(d1, grad_a2)
            grad_h1 = model['drop1'].backward(h1, grad_d1)
            grad_a1 = model['nonlinear1'].backward(a1, grad_h1)
            model = miniBatchGradientDescent(model, momentum, _alpha, _learning_rate)

        for i in range(int(np.floor(N_train / minibatch_size))):

            x, y = trainSet.get_example(np.arange(i * minibatch_size, (i + 1) * minibatch_size))

            a1 = model['L1'].forward(x)
            h1 = model['nonlinear1'].forward(a1)
            d1 = model['drop1'].forward(h1, is_train = False)
            a2 = model['L2'].forward(d1)
            loss = model['loss'].forward(a2, y)
            train_loss += loss
            train_acc += np.sum(predict_label(a2) == y)
            train_count += len(y)

        train_acc = train_acc / train_count
        train_acc_record.append(train_acc)
        train_loss_record.append(train_loss)

        print('Training loss at epoch ' + str(t + 1) + ' is ' + str(train_loss))
        print('Training accuracy at epoch ' + str(t + 1) + ' is ' + str(train_acc))

        for i in range(int(np.floor(N_val / minibatch_size))):

            x, y = valSet.get_example(np.arange(i * minibatch_size, (i + 1) * minibatch_size))

            a1 = model['L1'].forward(x)
            h1 = model['nonlinear1'].forward(a1)
            d1 = model['drop1'].forward(h1, is_train = False)
            a2 = model['L2'].forward(d1)
            loss = model['loss'].forward(a2, y)
            val_loss += loss
            val_acc += np.sum(predict_label(a2) == y)
            val_count += len(y)

        val_loss_record.append(val_loss)
        val_acc = val_acc / val_count
        val_acc_record.append(val_acc)

        print('Validation accuracy at epoch ' + str(t + 1) + ' is ' + str(val_acc))

    json.dump({'train': train_acc_record, 'val': val_acc_record},
              open('MLP_lr' + str(main_params['learning_rate']) +
                   '_m' + str(main_params['alpha']) +
                   '_d' + str(main_params['dropout_rate']) +
                   '_a' + str(main_params['activation']) +
                   '.json', 'w'))

    print('Finish running!')
    return train_loss_record, val_loss_record



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', default=42)
    parser.add_argument('--learning_rate', default=0.01)
    parser.add_argument('--alpha', default=0.0)
    parser.add_argument('--dropout_rate', default=0.5)
    parser.add_argument('--num_epoch', default=10)
    parser.add_argument('--minibatch_size', default=5)
    parser.add_argument('--activation', default='relu')
    parser.add_argument('--input_file', default='mnist_subset.json')
    args = parser.parse_args()
    main_params = vars(args)
    main(main_params)