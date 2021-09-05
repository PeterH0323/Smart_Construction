# -*- coding: utf-8 -*-
# @Time    : 2020/7/31 21:10
# @Author  : PeterH
# @Email   : peterhuang0323@outlook.com
# @File    : merge_data.py
# @Software: PyCharm
# @Brief   : 将自己的数据集使用 yolov5 检测出人体，合入自己数据集生成的安全帽标签
# 使用命令 python detect.py --save-txt --source 图片路径， 可以在 output 中看到每个图片的 txt 标签文件

import os
from pathlib import Path
from tqdm import tqdm

YOLOV5_LABEL_ROOT = Path(r"E:\AI_Project\Smart_Construction_Project\inference\output")  # yolov5 导出的推理图片的 txt
DATASET_LABEL_ROOT = Path(r"E:\AI_Project\AI_Learning\Dataset\VOC2028\Labels")  # 数据集的路径

if __name__ == '__main__':

    print("Merging data, pls wait...")
    with tqdm(len(os.listdir(YOLOV5_LABEL_ROOT))) as p_bar:
        # 遍历文件里面有 .txt 结尾的
        for file_path in YOLOV5_LABEL_ROOT.iterdir():
            p_bar.update(1)
            # 判断 txt 文件才进行读取
            if not file_path.suffix == ".txt":
                continue

            with open(file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    # 只需要提取 0 -> person 的数据
                    if line.split()[0] != '0':
                        continue

                    data_path = DATASET_LABEL_ROOT.joinpath(file_path.name)
                    print(data_path)
                    # 汇总到数据集的标注文件
                    with open(data_path, "a") as fd:
                        fd.write(line)

    print("Merging data done! ")
