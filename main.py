import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

if __name__ == "__main__":
    X, Y, X_t, Y_t = read_data("./601666.csv")
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
    plt.show()
    plt.savefig("fit_result.jpg")
 