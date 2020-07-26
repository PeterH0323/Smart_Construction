# Smart_Construction
该项目是使用 YOLOv5 来训练在智能工地安全领域中头盔目标检测的应用

![](https://github.com/PeterH0323/Smart_Construction/blob/master/doc/output_1.png?raw=true)
![](https://github.com/PeterH0323/Smart_Construction/blob/master/doc/output_2.jpg?raw=true)
![](https://github.com/PeterH0323/Smart_Construction/blob/master/doc/output_3.jpg?raw=true)
![](https://github.com/PeterH0323/Smart_Construction/blob/master/doc/output_4.png?raw=true)

# YOLO v5训练自己数据集教程
使用的数据集：[Safety-Helmet-Wearing-Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset) ，感谢这位大神的开源数据集！
> 本文结合 [YOLOv5官方教程](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) 来写

## 环境准备
首先确保自己的环境：
```text
    Python >= 3.7
    Pytorch >= 1.5
```

## 训练自己的数据

### 1.创建自己的数据集配置文件

因为我这里只是判断 【人没有带头盔】、【人有带头盔】两种情况，基于 `data/coco128.yaml` 文件，创建自己的数据集配置文件 `custom_data.yaml`

```yaml

# 训练集和验证集的 labels 和 image 文件的位置
train: ./score/images/train
val: ./score/images/val

# number of classes
nc: 2

# class names
names: ['person', 'hat']
```

### 2. 创建每个图片对应的标签文件

使用标注工具类似于 [Labelbox](https://labelbox.com/) 、[CVAT](https://github.com/opencv/cvat) 、[精灵标注助手](http://www.jinglingbiaozhu.com/) 标注之后，需要生成每个图片对应的 `.txt` 文件，其规范如下：
- 每一行都是一个目标
- 类别序号是零索引开始的（从0开始）
- 每一行的坐标 `class x_center y_center width height` 格式
- 框坐标必须采用**归一化的 xywh**格式（从0到1）。如果您的框以像素为单位，则将`x_center`和`width`除以图像宽度，将`y_center`和`height`除以图像高度。代码如下：
```python
def convert(size, box):
    """
    将标注的 xml 文件生成的【左上角x,左上角y,右下角x，右下角y】标注转换为yolov5训练的坐标
    :param size: 图片的尺寸： [w,h]
    :param box: anchor box 的坐标 [左上角x,左上角y,右下角x,右下角y,]
    :return: 转换后的 [x,y,w,h]
    """

    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    dw = np.float32(1. / int(size[0]))
    dh = np.float32(1. / int(size[1]))

    w = x2 - x1
    h = y2 - y1
    x = x1 + (w / 2)
    y = y1 + (h / 2)

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]
```

生成的 `.txt` 文件放置的名字是图片的名字，放置在 label 文件夹中，例如：
```text
./score/images/train/00001.jpg  # image
./score/labels/train/00001.txt  # label
```

生成的 `.txt` 例子：
```text
1 0.1830000086920336 0.1396396430209279 0.13400000636465847 0.15915916301310062
1 0.5240000248886645 0.29129129834473133 0.0800000037997961 0.16816817224025726
1 0.6060000287834555 0.29579580295830965 0.08400000398978591 0.1771771814674139
1 0.6760000321082771 0.25375375989824533 0.10000000474974513 0.21321321837604046
0 0.39300001866649836 0.2552552614361048 0.17800000845454633 0.2822822891175747
0 0.7200000341981649 0.5570570705458522 0.25200001196935773 0.4294294398277998
0 0.7720000366680324 0.2567567629739642 0.1520000072196126 0.23123123683035374
```

### 3.文件放置规范
文件树如下

![](https://github.com/PeterH0323/Smart_Construction/blob/master/doc/File_tree.png?raw=true)

### 4. 选择一个你需要的模型
在文件夹 `./models` 下选择一个你需要的模型

### 5. 开始训练
这里选择了 `yolov5s` 模型进行训练，权重也是基于 `yolov5s.pt` 来训练

```python
$ python train.py --img 640 --batch 16 --epochs 5 --data ./data/custom_data.yaml --cfg ./models/yolov5s.yaml --weights ./yolov5s.pt
```

其中，`yolov5s.pt` 需要自行下载放在本工程的根目录即可，下载地址 [官方权重](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J)

### 6.看训练之后的结果
训练之后，会保存在 `./runs` 文件夹里面，每个 `exp` 文件里面可以看到训练的效果
![](https://github.com/PeterH0323/Smart_Construction/blob/master/doc/test_batch0_gt.jpg?raw=true)

# 侦测
侦测图片会保存在 `./inferenct/output/` 文件夹下

运行命令：
```shell script
$ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp stream
                            http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8  # http stream
```
