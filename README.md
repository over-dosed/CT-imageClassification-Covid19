## WeatherRecognition
a simple network to recognize weather form a picture
# To get the dataset:
https://www.kaggle.com/jehanbhathena/weather-dataset
# 数据集介绍
数据集由11个天气类，共6862张大小不一的天气图片组成
The pictures are divided into 11 classes: dew, fog/smog, frost, glaze, hail, lightning , rain, rainbow, rime, sandstorm and snow.
# 数据集处理
为处理方便，我采用图片下采样处理初始数据集，并将处理后的图片进行分割为训练集train,验证集val；以txt形式存储便于读取

## 天气预测失败，模型无法收敛，cost无法下降。转而使用CT图像预测新冠肺炎

# To get the dataset:
http://ictcf.biocuckoo.cn/

# 数据集介绍
进行ct图像预测，采用华科的数据集
数据集由3类ct图像组成
其中，训练图像共17714张，测试图片共1969张，包括nCT(negative)阴性,NiCT(no information) 以及 pCT(positive)阳性

# 数据预处理与读入
利用脚本文件，将图片全部下采样resize为64*64大小，8位位深度的图片，同时制作txt文件，将path和label按行写入txt

读取时按行读入txt，读取图片数据拼接至data,label数据也拼接至label的data

np的reshape和T转置函数，得到
训练数据向量shape为：（64*64*3，17714） ，训练label向量shape为：（3，17714） // 3为类别，17714是样本数目

# 模型设置
具有一个大小为30的隐藏层的DNN全连接层线性神经网络模型

layers_dims = [64*64*3,30,3] 
learing rate = 0.075           // 模型超参数

（64*64*3，17714）=>   w1:(64*64*3,30) => (30,17714) => Relu   // 输入层 => 隐藏层
 (30,17714) => w2:(30,3) => (3,17714) => Softmax               // 隐藏层 => 输出层

 # 模型训练与结果
 设备： 搭载intel 9750h 电脑，16GB Ram内存
 训练3000 epoch ，cost 与 accuracy 结果如下：

 cost after iteration 2999: -0.670973486088058
 train accuracy = 0.7724398780625494
 test  accuracy = 0.7694261046216353

 # 实验结束