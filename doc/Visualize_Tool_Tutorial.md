# 如何使用可视化工具

本文为可视化界面的使用教程

![](./visualize_tool_gif_v2_1.gif)

## 0. 前期准备
将 权重文件 放到 `weights` 文件夹中，确保**有且只有一个** `.pt` 文件；

执行代码，运行可视化界面
```shell script
python visual_interface.py
```

**注意**：开始的时候程序会去加载模型，需要大概等待`1~3`秒左右的时间，加载成功后，请确认中间彩色进度条下面的 `Using weight：` 是否是你的权重文件名

## 一、导入

点击按钮 `Import` 按钮，选择 `视频` 或者 `图片` 文件，点击确定

**注意**：
1) 如果视频导入之后无法 显示 or 播放，请下载并安装`LAV 解码器`， 下载链接：[LAV 解码器](https://files.1f0.de/lavf/LAVFilters-0.73.1.exe)
2) 导入图片文件的话，按钮 `Paly` 和 `Pause` 会被失能

## 二、进行推理

点击按钮 `Predict` 进行推理，等待进度条跑完，进度条隔壁会显示目前推理的 `FPS` 指标

**注意**：
1) 在推理过程中，所有的按钮都会被**失能**；
2) 如果你的实时显示推理过程会导致软件卡死，可以取消掉位于 `Predict` 按钮上方的 `Real Time Predict`的`√`，或者请将 `visual_interface.py` 中的 `real_time_show_predict` 改为 `False`

## 三、对推理视频进行播放

点击按钮 `Play` 进行播放，此时【原视频】和【推理视频】**同时**播放，点击 `Pause` 即可【暂停】

## 四、打开推理文件夹输出路径

点击按钮`Open in Browser`，使用文件浏览器打开推理文件输出位置

## 五、推理过程

您可以在 `Predict info` 中实时查看推理的过程

## 六、GPU 信息

您可以在最下面的折线图观察到 `GPU` 的实时使用率的变化情况

## 七. 打包 exe 文件

1. 执行命令进行打包（带有调试信息的）
```shell script
pyinstaller -D -c --icon=./UI/icon/icon.ico visual_interface.py
```

或者， 执行命令进行打包（无调试信息）
```shell script
pyinstaller -D -w --icon=./UI/icon/icon.ico visual_interface.py
```

2. 等待打包完成
3. 打包完成后，生成的 `exe` 位于 `dist` 文件夹中的 `visual_interface` 中
4. 将 `weights`文件夹 放到 `dist` 文件夹中的 `visual_interface` 中，并确保 `weights`文件夹 中**有且只有一个**您需要的权重文件
5. 将 `dist`中的 `model` 和 `UI`文件夹 放到 `dist` 文件夹中的 `visual_interface` 中
6. 进入`visual_interface` 文件夹，双击 `exe` 执行程序
7. Enjoy !

