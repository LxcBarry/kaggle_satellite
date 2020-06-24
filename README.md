# Kaggle satellite 

## reference
[类似比赛冠军描述](https://blog.csdn.net/weixin_34265814/article/details/89834008)  

## 先把训练的框标出来
[代码](code/view_train.py)

## 发现一篇提供工具的kernel
[Andrew Lukyanenko](https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools)   
先跟着思路做一下  

### 安装工具库
```shell
# this is a great library for image augmentation which makes it easier and more convenient
pip install albumentations 

# this is a great library which makes using PyTorch easier, helps with reprodicibility and contains a lot of useful utils
pip install catalyst

# this is a great library with convenient wrappers for models, losses and other useful things
pip install segmentation_models_pytorch

# this is a great library with many useful shortcuts for building pytorch models
pip install pytorch-toolbelt


```

里面提供了一些有用的工具函数，这里借用了  
[code](code/helper_function.py)  

### 数据预处理
>create a list of unique image ids and the count of 
masks for images. This will allow us to make a stratified 
split based on this count.  

意思就是，用不同的掩膜做id标记每张图片，只要有效的图片作为训练数据  
![1](code/img/1.png)  
```python
id_mask_count = train.loc[train['EncodedPixels'].isnull() == False,'Image_Label'].\
    apply(lambda x:x.split('_')[0]).value_counts().\
    reset_index().rename(columns={'index':'img_id','Image_Label':'count'})
```



```python
get_img(img_name) # 获取图片，需要设置全局变量path
make_mask(img_name) # 获取掩膜

# 利用albu做数据扩增
# 几个我觉得能用上的函数

albu.HorizontalFlip(p=0.5)
albu.GridDistortion(p=0.5)
albu.Resize(320, 640)
albu.RandomRotate90(p=1)
albu.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)

# 这几个函数都可以使用下面函数得到扩增
get_training_augmentation()

```
## 5折交叉验证
todo

## 更新
**2019/9/23**
训练出第一个模型resnet50+Unet best model:dice = 0.5288  
开始做预测与后处理部分  
  
**2019/10/13**  
上传了8次submission,最佳dice:0.648  
开始做k折交叉验证

 

**2019/12/30**  
比较遗憾，由于时间关系，没能优化，最后比赛结果也不是很好