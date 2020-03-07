# personReco

## 1.1、行人检测模型

​	行人检测采用yolov3，使用Darknet53网络。

``` python
############# 行人检测模型初始化 #############
model = Darknet(cfg, img_size)    # yolov3使用Darknet53网络
```



## 1.2、行人重识别模型

​	行人重识别模型使用ResNet50网络。

``` python
############# 行人重识别模型初始化 #############
query_loader, num_query = make_data_loader(reidCfg)    # 迭代器，待查样本数
print("query_loader:", type(query_loader), query_loader)

reidModel = build_model(reidCfg, num_classes=10126)    # 行人重识别使用ResNet50网络
reidModel.load_param(reidCfg.TEST.WEIGHT)
reidModel.to(device).eval()
```

