# Multimodel Sentiment Analysis
当代人工智能第五次实验 多模态情感分析

## Setup
基于python3.8和CUDA，pytorch

下列库是比较重要的
- numpy
- pandas
- sklearn
- torch
- transformers
- torchvision
- PIL

可以通过运行下列的命令行安装必备库

```python
pip install -r requirements.txt
```

## Repository Structure
```python
|-- data #本次实验的匿名数据集
    |-- xxx.jpg #图像数据  
    |-- xxx.txt #文本数据
|-- README.md #此文件
|-- main.py #主要文件
|-- models.py #包含了两种消融模型两种混合模型的类
|-- requirement.txt
|-- test_without_label(原文件).txt #原有的测试集索引文件
|-- test_without_label.txt #作业所要求提交的测试集预测文件，由于作业要求需要再原文件上修改
|-- train.py #训练和预测的类
|-- train.txt #训练集索引文件
|-- ***bert-base-uncased #未实际导入github的文件夹，bert-base-model的预训练模型
|-- ***resnet #未实际导入github的文件夹，resnet的预训练模型
```
## Code Running
数据集，预训练模型我都打包发往助教的邮箱，请助教查收！

如何运行文件？
可以直接运行 python main.py，有三个args，--model，--fusion， --epoch
--model取值为 weight, concat ， 分别代表权重模型和拼接模型
--fusion取值为 text_and_image, text, image 当仅当取值为text_and_image时，model参数才生效，它们分别代表融合模型，文本消融模型和图像消融模型。
--epoch 为int 型，默认值为6， 建议测试的时候不要超过5，会过拟合

前两者默认取值分别为weight和text_and_image

另外的，为了防止后续生成的文件更改所递交的预测文件，即时的预测生成文件并非test_without_label.txt，而是predict1.txt。正确率评判标准以test_without_label.txt评判！！！

##References
- https://github.com/abhimishra91/transformers-tutorials
- https://github.com/liyunfan1223/multimodal-sentiment-analysis
