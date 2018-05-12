import numpy as np 
import matplotlib.pyplot as plt 

#the derivative func of the original func
def dJ(theta):
    return 2*(theta - 2.5)

#the original func
def J(theta):
    try:
        return (theta - 2.5) ** 2 - 1
    except:
        return float('inf')

def gradient_descent(initial_theta, eta, epsilon = 1e-8, n_iters = 1e4):
    theta = initial_theta
    i_iter = 0
    while i_iter < n_iters:
        gradient = dJ(theta)
        pre_theta = theta 
        theta -= eta * gradient
        theta_history.append(theta)
        i_iter += 1
        if abs(J(theta) - J(pre_theta)) < epsilon:
            break
    return theta

def plot_theta_history():
    plt.plot(plot_x, J(plot_x))
    plt.plot(np.array(theta_history), J(np.array(theta_history)), color = 'r', marker = '*')
    plt.show()

plot_x = np.linspace(-1, 6, 141)
plot_y = (plot_x - 2.5) ** 2 - 1
eta = 0.01
theta = 0.0
epsilon = 1e-8
theta_history= []
gradient_descent(theta, eta, epsilon)
plot_theta_history()
print(theta_history[-1])
