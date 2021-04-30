#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Time    : 2020/7/29 20:29
# @Author  : PeterH
# @Email   : peterhuang0323@outlook.com
# @File    : data_cfg.py
# @Software: PyCharm
# @Brief   : 生成测试、验证、训练的图片和标签


import argparse
import os
import pathlib
import shutil

import numpy as np

from PIL import Image, ImageDraw
from tqdm import tqdm
from xml.dom.minidom import parse


def convert_box(size, box):
    """
    将标注的 xml 文件标注转换为 darknet 形的坐标
    :param size: 图片的尺寸： [w,h]
    :param box: anchor box 的坐标 [左上角x,左上角y,右下角x,右下角y,]
    :return: 转换后的 [x,y,w,h]
    """
    box = np.asarray(box, dtype=float)
    box[2:] -= box[:2]
    box[:2] += box[2:] / 2
    box = box.reshape(-1, 2) / np.asarray(size, dtype=float)
    return box.reshape(-1).tolist()


def test_dataset_box_feature(file_path, boxes):
    """
    使用样本数据测试数据集的建议框
    :param file_path: 图片路径
    :param boxes: 全部的点 [[name, x1, y1, x2, y2]]
    :return: None
    """
    img = Image.open(file_path)
    img_draw = ImageDraw.Draw(img)
    for box in boxes:
        img_draw.rectangle(box[1:], outline='red')
    img.show()


def parse_annotation_data(annotation_path):
    """
    解析 xml 数据
    :param annotation_path: xml 文件路径
    :return: [w, h], [x1, y1, x2, y2]
    """
    xml_root = parse(annotation_path).documentElement
    img_size = xml_root.getElementsByTagName("size")[0]
    objects = xml_root.getElementsByTagName("object")
    size = [
        img_size.getElementsByTagName("width")[0].childNodes[0].data,
        img_size.getElementsByTagName("height")[0].childNodes[0].data
    ]
    img_boxes = [
        [
            box.getElementsByTagName("name")[0].childNodes[0].data,
            int(box.getElementsByTagName("xmin")[0].childNodes[0].data),
            int(box.getElementsByTagName("ymin")[0].childNodes[0].data),
            int(box.getElementsByTagName("xmax")[0].childNodes[0].data),
            int(box.getElementsByTagName("ymax")[0].childNodes[0].data),
        ]
        for box in objects
    ]
    return size, img_boxes


class VocToYolo(object):

    def __init__(self, voc_root, target_root):
        self.voc_root = pathlib.Path(voc_root)
        self.target_root = pathlib.Path(target_root)
        self.image_set_root = self.voc_root.joinpath('ImageSets', 'Main')  # 图片区分文件的路径
        self.jpg_image_root = self.voc_root.joinpath('JPEGImages')  # 图片的位置
        self.annotation_root = self.voc_root.joinpath('Annotations')  # 数据集标签文件的位置

        self.label_root = self.voc_root.joinpath('Labels')  # 进行归一化之后的标签位置
        self.label_map = {'person': 1, 'hat': 2}  # VOC 数据集与目标数据集类别映射关系

        self.target_image_root = self.target_root.joinpath('images')  # 区分训练集、测试集、验证集的图片目标路径
        self.target_label_root = self.target_root.joinpath('labels')  # 区分训练集、测试集、验证集的标签文件目标路径

    def save_label_file(self, file_name, size, boxes):
        """
        保存标签的解析文件
        :param file_name:
        :param size:
        :param boxes:
        """
        lines = [' '.join(map(str, [self.label_map[box[0]], *convert_box(size, box[1:])])) for box in boxes if box[0] in self.label_map]
        with open(self.label_root.joinpath(file_name).with_suffix('.txt'), mode='w') as f:
            f.write('\n'.join(lines))

    def copy_data(self, dataset_type):
        """
        :param dataset_type: 生成数据集的种类
        :return:
        """

        target_image_root = self.target_image_root.joinpath(dataset_type)
        target_label_root = self.target_label_root.joinpath(dataset_type)

        target_image_root.mkdir(parents=True, exist_ok=True)
        target_label_root.mkdir(parents=True, exist_ok=True)

        with open(self.image_set_root.joinpath(dataset_type).with_suffix(".txt"), encoding="UTF-8") as f:
            lines = f.read().splitlines()
        for img_name in tqdm(lines):
            # 复制图片
            src = self.jpg_image_root.joinpath(img_name).with_suffix('.jpg')
            dst = target_image_root.joinpath(img_name).with_suffix('.jpg')
            shutil.copyfile(src, dst)
            # 复制 label
            src = self.label_root.joinpath(img_name).with_suffix('.txt')
            dst = target_label_root.joinpath(img_name).with_suffix('.txt')
            shutil.copyfile(src, dst)

    def run(self):
        # 清空标签文件夹
        if self.label_root.exists():
            shutil.rmtree(self.label_root)
        self.label_root.mkdir(exist_ok=True)

        # 生成标签
        with tqdm(total=len(os.listdir(self.annotation_root))) as p_bar:
            for file in self.annotation_root.iterdir():
                size, img_boxes = parse_annotation_data(str(file))
                self.save_label_file(file.name, size, img_boxes)
                p_bar.update(1)

        # 将文件进行 train、val、test 的区分
        for dataset_input_type in ["train", "val", "test"]:
            print(f"Copying data {dataset_input_type}, pls wait...")
            self.copy_data(dataset_input_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc-root', type=str, default='../datasets/VOC2028')
    parser.add_argument('--target-root', type=str, default='../datasets/person-head-helmet')
    VocToYolo(**vars(parser.parse_args())).run()
