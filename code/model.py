import time
import numpy as np
from data_loader import *
from DNN_functions import *

training_data, test_data = load_data() #导入数据集

#导入数据分离
train_X = training_data[0].reshape((config.trainNum,-1)).T
train_Y = training_data[1].T

test_X = test_data[0].reshape((config.valNum,-1)).T
test_Y = test_data[1].T


n_x = train_X.shape[0]
n_y = train_X.shape[1]
layers_dims = [n_x,50,30,20,10,config.numClass]  #4个大小为 50、30、20、10的隐藏层
learning_rate = 0.0075

def L_layer_model(X, Y, layers_dims, learning_rate = 0.01, num_iterations = 3000, print_cost=False):
    
    np.random.seed(1)
    costs = []      
    #parameters = initialize_parameters_deep(layers_dims) #随机初始化参数
    parameters = np.load('./model/DNN3000-1.npy', allow_pickle=True).item()
    start = time.time()
    thisTime = start

    for i in range(0, num_iterations):                                     #进行迭代
        
        AL,caches = L_model_forward(X, parameters)                         #一次向前传播
        cost = computer_cost_softmax(AL, Y)                                         #计算cost
        grads = L_model_backward(AL, Y, caches)                            #一次向后传播，计算梯度
        parameters = update_parameters(parameters, grads, learning_rate)   #根据梯度更新一次系数
        
        if print_cost and i % 20 == 0 or i == num_iterations - 1:

            elapsed = (time.time() - thisTime)
            thisTime = time.time()
            elapsed_all = (time.time() - start)

            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            print("this time：" + str(elapsed) + "  /  all time: " + str(elapsed_all))
            print("train accuracy = " + str(predict(AL,train_Y)))

        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)
    
    return parameters, costs

#进行一次三千迭代的训练
parameters, costs = L_layer_model(train_X, train_Y, layers_dims, num_iterations = 400, print_cost = True)

#保存训练的参数

np.save('./model/DNN3000-1.npy',parameters)

#parameters = np.load('./model/DNN3000.npy', allow_pickle=True).item()

#将训练好的参数进行测试
AL,caches = L_model_forward(train_X,parameters)
print("train accuracy = " + str(predict(AL,train_Y)))

AL,caches = L_model_forward(test_X,parameters)
print("test accuracy = " + str(predict(AL,test_Y)))
