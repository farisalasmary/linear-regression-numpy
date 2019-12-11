
import numpy as np
import pandas as pd
from numpy.linalg import inv, norm



def normalize(X):
    mu = X.mean(0)
    std = X.std(0)
    return (X - mu) / std, mu, std


def add_bias_column(X):
    ones = np.ones(X.shape[0]).reshape(X.shape[0], 1)
    return np.concatenate((ones, X), 1)


def normal_equation(X, y):
    w = inv(X.T.dot(X)).dot(X.T).dot(y)
    return w


def compute_error(X, y, w):
    return (y - X.dot(w)).T.dot(y - X.dot(w))



def gradient_descent(X, y, alpha=0.001, epsilon=0.0001):
    m, n = X.shape
    w_old = np.random.rand(n).reshape(n, 1)
    w_new = np.zeros((n, 1))
    i = 0
    while norm(w_new - w_old)  > epsilon:
        w_old = w_new
        RSS = compute_error(X, y, w_old)
        grad_RSS = (X.T.dot(X).dot(w_old) - X.T.dot(y))   
        w_new = w_old - alpha * grad_RSS
        print(i, np.squeeze(RSS))
        i += 1
    return w_new


if __name__ == '__main__':
    train_data = pd.read_csv('kc_house_train_data.csv')
    test_data = pd.read_csv('kc_house_test_data.csv')
    
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot']
    target = 'price'
    
    X = train_data[features].values
    y = train_data[target].values.reshape(X.shape[0], 1)
    
    X, mu, std = normalize(X)
    X = add_bias_column(X)
    
    ne_w = normal_equation(X, y)
    gd_w = gradient_descent(X, y, alpha=0.00001)
    
    print('-'*70)
    print('Normal Equation Model:\n', ne_w, '\n', '-'*70)
    print('Gradient Descent Model:\n', gd_w, '\n', '-'*70)
    print('Difference between them:\n', ne_w - gd_w, '\n', '-'*70)
    
