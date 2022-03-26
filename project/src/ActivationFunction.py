import numpy as np

#sigmoid激活函数用于处理线性无法处理的问题
def sigmoid(x,deriv=False):
    if deriv == True:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))
