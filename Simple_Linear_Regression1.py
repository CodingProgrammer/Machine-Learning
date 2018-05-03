import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegression1:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        '''根据训练集训练Simple Linear Regression模型'''
        assert x_train.ndim == 1, \
            "Simple Linear Regression can only solve single feature"
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0 #分子
        d = 0.0 #分母
        for x, y in zip(x_train, y_train):
            num += (x - x_mean) * (y - y_mean)
            d += (x - x_mean) ** 2
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        print('Fitted!')
        return self
    
    def predict(self, x_predict):
        '''给定预测数据集x_predict，返回x_predict的结果向量'''
        assert x_predict.ndim == 1, \
            "Simple Linear Regression can only solve single feature"
        assert self.a_ and self.b_, \
            "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        '''给定单个预测数据x_predict，返回x_predict的结果值'''
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return 'SimpleLinearRegression1()'    



x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])
x_predict = [6, 7, 8]
reg1 = SimpleLinearRegression1()
reg1.fit(x, y)
print(reg1.predict(np.array(x_predict)))
print(reg1.a_)
print(reg1.b_)
y_hat1 = reg1.predict(x)
plt.scatter(x, y)
plt.plot(x, y_hat1, color = 'r')
plt.axis([0, 6, 0, 6])
plt.show()