# WeatherRecognition
a simple network to recognize weather form a picture

# To get the dataset:

https://www.kaggle.com/jehanbhathena/weather-dataset

# 数据集介绍
数据集由11个天气类，共6862张大小不一的天气图片组成
The pictures are divided into 11 classes: dew, fog/smog, frost, glaze, hail, lightning , rain, rainbow, rime, sandstorm and snow.

# 数据集处理
为处理方便，我采用图片下采样处理初始数据集，并将处理后的图片进行分割为训练集train,验证集val；以txt形式存储便于读取