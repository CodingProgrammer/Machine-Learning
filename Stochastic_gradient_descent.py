import numpy as np 
import matplotlib.pyplot as plt 
def J(theta, X_b, y):
    try:
        return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
    except:
        return float('inf')

def dJ_sgd(theta, X_b_i, y_i):
    return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2.     

def sgd(X_b, y,  initial_theta, n_iters):
    t0 =5
    t1 =50

    def learing_rate(t):
        return t0 / (t + t1)

    theta = initial_theta
    for cur_iter in range(n_iters):
        rand_i = np.random.randint(len(X_b))
        gradient = dJ_sgd(theta, X_b[rand_i], y[rand_i])
        theta = theta - learing_rate(cur_iter) * gradient
    
    return theta


m = 100000
x = np.random.normal(size = m)
X = x.reshape(-1, 1)
y = 4. * x + 3. + np.random.normal(0, 3, size = m)

X_b = np.hstack([np.ones((len(X), 1)), X])
initial_theta = np.zeros(X_b.shape[1])
eta = 0.01

theta = sgd(X_b, y, initial_theta, n_iters = len(X_b) // 3)

print(theta)
