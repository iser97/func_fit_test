import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sst
from scipy.optimize import curve_fit

#用于计算Lxx, Lyy
def laa(x):
    x_mean = np.mean(x)
    lxx = np.sum((x-x_mean)**2)
    return lxx

#用于计算Lxy
def lab(x,y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    lxy = np.sum((x-x_mean)*(y-y_mean))
    return lxy

def read_data(data_root):
    ''' read 601666.csv file and preprocess this file
        Args:
            data_root: the data file root
    '''
    data = pd.read_csv(data_root, encoding="gbk")
    X = []
    Y = []
    for index, line in data.iterrows():
        line = dict(line)
        Y_t = line["收盘价"]
        X_t = line["收盘价.1"]
        X.append(X_t)
        Y.append(Y_t)
    X_t_t_1 = [X[index]/X[index-1] for index in range(1, len(X), 1)]  # 将X的时间序列转换为X_t/X_t-1的时间序列
    Y_t_t_1 = [Y[index]/Y[index-1] for index in range(1, len(Y), 1)]  # 将Y的时间序列转换为Y_t/Y_t-1的时间序列
    assert(len(X_t_t_1) == len(Y_t_t_1))
    # X = np.log(np.array(X))
    # Y = np.log(np.array(Y))
    X_t_t_1 = np.log(np.array(X_t_t_1))
    Y_t_t_1 = np.log(np.array(Y_t_t_1))
    return X, Y, X_t_t_1, Y_t_t_1

def func(x, alpha, beta):
    return alpha + beta*x

def question_one(X_t, Y_t):
    popt, pcov = curve_fit(func, X_t, Y_t)
    alpha = popt[0]
    beta = popt[1]
    print("alpha = ", alpha)
    print("beta = ", beta)
    yvals = func(X_t, alpha, beta)
    plot1 = plt.plot(X_t, Y_t, "s", label="original values")
    plot2 = plt.plot(X_t, yvals, "r", label="fit values")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=4) #指定legend的位置右下角
    plt.title('curve_fit')
    plt.savefig("fit_result.jpg")
    # plt.show()
    return alpha, beta

def question_two(x, y, alpha=0.05):
    '''
    Args:
        alpha: 是置信区间的取值，0.05则是95%的置信区间
    '''
    n = len(x)
    lxx = laa(x)
    lyy = laa(y)
    lxy = lab(x, y)

    R = lxy/(np.sqrt(lxx) * np.sqrt(lyy))
    R2 = R*R   #计算相关系数与决定系数

    b_est = lxy/lxx  #计算b估计
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    a_est = y_mean - b_est * x_mean   #计算a估计
    Qe = lyy - b_est * lxy
    sigma_est2 = Qe / (n - 2)

    sigma_est = np.sqrt(sigma_est2) #sigma估计

    test = np.abs(b_est * np.sqrt(lxx))/sigma_est
    test_level = sst.t.ppf(1 - alpha/2, df=n - 2)
    linear_test = test > test_level   #线性回归检验

    #a,b的置信区间
    b_int = [b_est - test_level * sigma_est / np.sqrt(lxx), b_est + test_level * sigma_est / np.sqrt(lxx)]
    a_int = [y_mean - b_int[1] * x_mean, y_mean - b_int[0] * x_mean]

    return a_est, b_est, a_int, b_int


if __name__ == "__main__":
    X, Y, X_t, Y_t = read_data("./601666.csv")
    alpha, beta = question_one(X_t, Y_t)
    print("回归拟合得到的alpha=", alpha, "  beta=", beta)
    alpha, beta, alpha_int, beta_int = question_two(X_t, Y_t, alpha=0.05)
    print("alpha的统计量=", alpha, "   beta的统计量=", beta, "alpha的置信区间：", alpha_int,  "    beta的置信区间：", beta_int)
    
 