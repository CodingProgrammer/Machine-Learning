import numpy as np 
import matplotlib.pyplot as plt 
def J(theta, X_b, y):
    try:
        return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
    except:
        return float('inf')

def dJ(theta, X_b, y):
    return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(y)

def gradient_descent(X_b, y, initial_theta, eta, n_itera = 1e4, epsilon = 1e-8):
    theta = initial_theta
    cur_iter = 0

    while cur_iter < n_itera:
        gradient = dJ(theta, X_b, y)
        last_theta = theta  
        theta = theta - eta * gradient
        if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
            break
        cur_iter += 1

    return theta

m = 100000
x = np.random.normal(size = m)
X = x.reshape(-1, 1)
y = 4. * x + 3. + np.random.normal(0, 3, size = m)

X_b = np.hstack([np.ones((len(X), 1)), X])
initial_theta = np.zeros(X_b.shape[1])
eta = 0.01

theta = gradient_descent(X_b, y, initial_theta, eta)

print(theta)
