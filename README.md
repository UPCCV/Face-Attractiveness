## Face Attriactiveness by keras

基于深度学习的颜值评价

本文基于[SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)数据库实现基于深度学习的颜值评价

## 下载数据

SCUT-FBP5500是包含5500张人脸图片的数据集，其下载地址为[百度网盘](https://pan.baidu.com/s/1Ff2W2VLJ1ZbWSeV5JbF0Iw)访问密码: if7p ，Size = 172MB

![](https://raw.githubusercontent.com/HCIILAB/SCUT-FBP5500-Database-Release/master/SCUT-FBP5500.jpg)

## 训练

[基于深度学习的颜值打分器](https://zhuanlan.zhihu.com/p/36138077)曾实现过一个基于ResNet50的网络进行打分，其存在以下两个问题：

1. 所有数据一次性读入内存，小于20G的机器无法训练

2. 选用模型太大，运行时需要的显存在8G以上，一般的机器也无法运行

针对以上两个问题，本文采用fit_generator，大大减少了训练所需的资源,模型也换成自己的小模型，取得了不错的效果.

## 部署

	cd web
	python app.py

打开浏览器,输入网址localhost:5000.选择一张图片上传，稍等片刻就会返回识别结果。

## 参考

[基于深度学习的颜值打分器](https://zhuanlan.zhihu.com/p/36138077)