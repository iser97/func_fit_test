# 参数拟合

题目是：
![题目](./题目.png)

将 $ln(Y_t/Y_{t-1})$ 与 $ln(X_t/X_{t-1})$ 分别当做因变量与自变量，因此需要先读取数据，计算对应的$ln(Y_t/Y_{t-1})$ 与 $ln(X_t/X_{t-1})$ 当做新的Y与X， 因此需要拟合的函数就是$Y = \alpha * X + \beta$， $\mu_t$应该是拟合误差项
