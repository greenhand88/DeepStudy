import src.ActivationFunction as acFunction
import numpy as np
class MyNetwork(object):#bp网络
    def __init__(self):
        self.w1 = 2 * np.random.random((4, 5)) - 1  #第一层权重矩阵:4*5    2是标准差,-1是期望
        self.w2 = 2 * np.random.random((5, 1)) - 1  #第二层权重矩阵:5*1    2是标准差,-1是期望

    def forward_prop(self, input_data):#向前传播
        self.l0 = input_data    #第0层输入
        self.l1 = acFunction.sigmoid(np.array(np.dot(self.l0, self.w1), dtype=np.float32))  # 第一层对输入进行矩阵乘法后变成数组再经过激活函数处理
        self.l2 = acFunction.sigmoid(np.array(np.dot(self.l1, self.w2), dtype=np.float32))  # 第二层对输入进行矩阵乘法后变成数组再经过激活函数处理
        l2_error = label[st_idx:ed_idx] - self.l2  #标签减去模型处理的结果
        return self.l2, l2_error

    def backward_prop(self, output_error):#反向传播
        l2_error = output_error
        l2_delta = l2_error * acFunction.sigmoid(self.l2, deriv=True)  # 梯度下降
        l1_delta = np.dot(l2_delta, self.w2.T) * acFunction.sigmoid(self.l1, deriv=True)  # 梯度下降
        self.w2 += lr * np.array(np.dot(self.l1.T, l2_delta), dtype=np.float32)#更新参数
        self.w1 += lr * np.array(np.dot(self.l0.T, l1_delta), dtype=np.float32)#更新参数

# data  :  [x0, x1, x2, x3]
# label :  x0 + x1 - x2 - x3 > 0? 1 : 0
def generate_data(data_size):
    data = np.empty((1, 4), dtype=object)#创建数组
    label = np.empty((1, 1), dtype=object)#创建数组
    for i in range(0, data_size):
        x = np.array([np.random.randint(0, 10, size=4)])#创建1*4的的数组
        res = int((x[0][0]+x[0][1] -x[0][2]-x[0][3])>0)#每组数据的标签
        y = np.array([[res]])
        data = np.concatenate((data, x), axis=0)#拼接数组
        label = np.concatenate((label, y), axis=0)#拼接数组
    return data[1:], label[1:]

if __name__ == '__main__':
    # size of data, batch size
    data_size = 100
    batch_sz = 10#一轮10次
    # learning rate, max iteration
    lr = 0.1
    max_iter = 5000#500轮的训练
    data, label = generate_data(data_size)
    NN = MyNetwork()
    for iters in range(max_iter):
        # starting index and ending index for fetching data
        st_idx = (iters % 10) * batch_sz
        ed_idx = st_idx + batch_sz
        input_data = data[st_idx : ed_idx]#选择输入的数据
        outcome, output_error = NN.forward_prop(input_data)
        if iters % 500 == 0:#每50轮打印一次loss值
            print("The loss is %f" % (np.mean(np.abs(output_error))))#对loss求均值并打印
            #print output_error.tolist()
            continue
        NN.backward_prop(output_error)