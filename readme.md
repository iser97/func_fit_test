# 参数拟合

题目是：
![题目](./题目.png)

将![](http://latex.codecogs.com/svg.latex?\ln(Y_t/Y_{t-1})) 与![](http://latex.codecogs.com/svg.latex?\ln(X_t/X_{t-1})) 分别当做因变量与自变量，因此需要先读取数据，计算对应的![](http://latex.codecogs.com/svg.latex?\ln(Y_t/Y_{t-1})) 与![](http://latex.codecogs.com/svg.latex?\ln(X_t/X_{t-1})) 当做新的Y与X，这部分通过函数read_data来完成，所以需要拟合的函数就是Y= alpha*X + Beta，$\mu_t$应该是拟合误差项
