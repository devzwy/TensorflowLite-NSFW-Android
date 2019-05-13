# TensorflowLite-NSFW-Android
一步一步安装TensorFlow、生成.tflite文件、移植android/iOS平台


### 安装TensorFlow

1. [安装Anaconda](https://www.anaconda.com/distribution/#macos)(安装3.7版本)

2. 安装成功后打开终端输入如下命令创建一个新的虚拟环境，方法是选择 Python 解析器并创建一个 ./venv 目录来存放它：
```Python
    conda create -n venv pip python=3.6  # select python version
```

3.激活虚拟环境：
```
source activate venv
```

4.安装TensorFlow

```
pip install --ignore-installed --upgrade tensorflow==1.13.1
```

5.[克隆open_nsfw项目](https://github.com/devzwy/NSFW-Python)


6.下载PyCharm编译器

7.导入open_nsfw的项目，修改检测图片绝对路径，运行项目可以得到类似如下值：  
![nsfw_img](https://github.com/devzwy/TensorflowLite-NSFW-Android/blob/master/img/nsfw_img.png)


8.复制nsfw.tflite文件到android项目

9.参考本demo

10.运行demo，获取类似如下结果值：  


![nsfw_img](https://github.com/devzwy/TensorflowLite-NSFW-Android/blob/master/img/aaaaa.png)
