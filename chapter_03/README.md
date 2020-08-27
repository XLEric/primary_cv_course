# 第3章  
## 章节介绍(Chapter Abstract)   
* 分类任务    
* 包括：  
    1）分类任务-数据集制作  
    2）分类任务-数据迭代器  
    3）模型训练  
    4）模型前向推断  

## 数据集（DataSets）  
* http://www.vision.caltech.edu/visipedia/CUB-200-2011.html  
  Wah C., Branson S., Welinder P., Perona P., Belongie S. “The Caltech-UCSD Birds-200-2011 Dataset.” Computation & Neural Systems Technical Report, CNS-TR-2011-001  

## 具体内容  

* 1、在 chapter_03 文件夹下新建文件夹datasets，或是命令行构建 mkdir datasets  
* 2、将下载数据集压缩包 CUB_200_2011.tgz 在datasets文件夹下解压。   
* 3、在 chapter_03 根目录下运行命令： python prepare_data/make_train_test_datasets.py ，制作数据集，会在 datasets 文件夹下生成 train_datasets（训练集），test_datasets（测试集）两个文件夹。制作数据的默认模式为目标crop标注边界框（bounding box 简称 bbox）区域，去除了背景，如下图bbox区域。   

  ![sample_1](https://github.com/XLEric/primary_cv_course/tree/master/chapter_03/samples/sample_1.png)  
  crop模式样本,如下图:    
  ![sample_2](https://github.com/XLEric/primary_cv_course/tree/master/chapter_03/samples/sample_2.png)  

* 4、模型训练，在 chapter_03 根目录下运行命令：python train.py  
  * A) 建议仔细阅读 train.py 接口参数注释。
  * B) 训练参数记录和模型保存路径会为 model_exp 文件夹，默认模式是每次训练都会清除之前的训练相关文件，可以设置 train.py 的 clear_model_exp = Flase，就可以保持每次训练相关文件不清除。

* 5、图片的前向推断运行命令：python inference.py , 可视化结果举例，如下图所示。

  ![sample_3](https://github.com/XLEric/primary_cv_course/tree/master/chapter_03/samples/sample_3.png)  


## 联系方式 （Contact）  
* E-mails: 305141918@qq.com   
