# -*- coding: utf-8 -*-
# @Time    : 2020/7/29 20:29
# @Author  : PeterH
# @Email   : peterhuang0323@outlook.com
# @File    : data_cfg.py
# @Software: PyCharm
# @Brief   : 生成测试、验证、训练的图片和标签

import os
import shutil
from pathlib import Path
from shutil import copyfile

from PIL import Image, ImageDraw
from xml.dom.minidom import parse
import numpy as np
from tqdm import tqdm

FILE_ROOT = Path(r"E:\AI_Project\AI_Learning\Dataset")

# 原始数据集
IMAGE_SET_ROOT = FILE_ROOT.joinpath(r"VOC2028\ImageSets\Main")  # 图片区分文件的路径
IMAGE_PATH = FILE_ROOT.joinpath(r"VOC2028\JPEGImages")  # 图片的位置
ANNOTATIONS_PATH = FILE_ROOT.joinpath(r"VOC2028\Annotations")  # 数据集标签文件的位置
LABELS_ROOT = FILE_ROOT.joinpath(r"VOC2028\Labels")  # 进行归一化之后的标签位置

# YOLO 需要的数据集形式的新数据集
DEST_IMAGES_PATH = Path(r"Safety_Helmet_Train_dataset\score\images")  # 区分训练集、测试集、验证集的图片目标路径
DEST_LABELS_PATH = Path(r"Safety_Helmet_Train_dataset\score\labels")  # 区分训练集、测试集、验证集的标签文件目标路径


def cord_converter(size, box):
    """
    将标注的 xml 文件标注转换为 darknet 形的坐标
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


def save_label_file(img_jpg_file_name, size, img_box):
    """
    保存标签的解析文件
    :param img_jpg_file_name:
    :param size:
    :param img_box:
    :return:
    """
    save_file_name = LABELS_ROOT.joinpath(img_jpg_file_name).with_suffix('.txt')
    with open(save_file_name, "a+") as f:
        for box in img_box:
            if box[0] == 'person':  # 数据集 xml 中的 person 指的是头
                cls_num = 1
            elif box[0] == 'hat':
                cls_num = 2
            else:
                continue
            new_box = cord_converter(size, box[1:])  # 转换坐标
            f.write(f"{cls_num} {new_box[0]} {new_box[1]} {new_box[2]} {new_box[3]}\n")


def test_dataset_box_feature(file_name, point_array):
    """
    使用样本数据测试数据集的建议框
    :param file_name: 图片文件名
    :param point_array: 全部的点 [建议框sx1,sy1,sx2,sy2]
    :return: None
    """
    im = Image.open(IMAGE_PATH.joinpath(file_name).with_suffix(".jpg"))
    im_draw = ImageDraw.Draw(im)
    for box in point_array:
        x1 = box[1]
        y1 = box[2]
        x2 = box[3]
        y2 = box[4]
        im_draw.rectangle((x1, y1, x2, y2), outline='red')

    im.show()


def get_xml_data(img_xml_file: Path):
    """
    获取 xml 数据
    :param img_xml_file: 图片路径
    :return:
    """
    dom = parse(str(img_xml_file))
    xml_root = dom.documentElement
    img_name = xml_root.getElementsByTagName("filename")[0].childNodes[0].data
    img_size = xml_root.getElementsByTagName("size")[0]
    objects = xml_root.getElementsByTagName("object")
    img_w = img_size.getElementsByTagName("width")[0].childNodes[0].data
    img_h = img_size.getElementsByTagName("height")[0].childNodes[0].data
    img_c = img_size.getElementsByTagName("depth")[0].childNodes[0].data
    img_box = []
    for box in objects:
        cls_name = box.getElementsByTagName("name")[0].childNodes[0].data
        x1 = int(box.getElementsByTagName("xmin")[0].childNodes[0].data)
        y1 = int(box.getElementsByTagName("ymin")[0].childNodes[0].data)
        x2 = int(box.getElementsByTagName("xmax")[0].childNodes[0].data)
        y2 = int(box.getElementsByTagName("ymax")[0].childNodes[0].data)
        img_box.append([cls_name, x1, y1, x2, y2])

    # test_dataset_box_feature(img_xml_file.name, img_box)
    save_label_file(img_xml_file.name, [img_w, img_h], img_box)


def copy_data(img_set_source, img_labels_root, imgs_source, dataset_type):
    """
    将标签文件和图片复制到最终数据集文件夹中
    :param img_set_source: 原数据集图片总路径
    :param img_labels_root: 生成的 txt 标签总路径
    :param imgs_source:
    :param dataset_type: 生成数据集的种类
    :return:
    """
    file_name = img_set_source.joinpath(dataset_type).with_suffix(".txt")  # 获取对应数据集种类的图片

    # 判断目标图片文件夹和标签文件夹是否存在，不存在则创建
    os.makedirs(FILE_ROOT.joinpath(DEST_IMAGES_PATH, dataset_type), exist_ok=True)
    os.makedirs(FILE_ROOT.joinpath(DEST_LABELS_PATH, dataset_type), exist_ok=True)

    with open(file_name, encoding="UTF-8") as f:
        for img_name in tqdm(f.read().splitlines()):
            img_sor_file = imgs_source.joinpath(img_name).with_suffix('.jpg')
            label_sor_file = img_labels_root.joinpath(img_name).with_suffix('.txt')

            # 复制图片
            dict_file = FILE_ROOT.joinpath(DEST_IMAGES_PATH, dataset_type, img_name).with_suffix('.jpg')
            copyfile(img_sor_file, dict_file)

            # 复制 label
            dict_file = FILE_ROOT.joinpath(DEST_LABELS_PATH, dataset_type, img_name).with_suffix('.txt')
            copyfile(label_sor_file, dict_file)


if __name__ == '__main__':
    root = ANNOTATIONS_PATH  # 数据集 xml 标签的位置

    if LABELS_ROOT.exists():
        # 清空标签文件夹
        print("Cleaning Label dir for safety generating label, pls wait...")
        shutil.rmtree(LABELS_ROOT)
        print("Cleaning Label dir done!")
    LABELS_ROOT.mkdir(exist_ok=True)  # 建立 Label 文件夹

    # 生成标签
    print("Generating Label files...")
    with tqdm(total=len(os.listdir(root))) as p_bar:
        for file in root.iterdir():
            p_bar.update(1)
            get_xml_data(file)

    # 将文件进行 train、val、test 的区分
    for dataset_input_type in ["train", "val", "test"]:
        print(f"Copying data {dataset_input_type}, pls wait...")
        copy_data(IMAGE_SET_ROOT, LABELS_ROOT, IMAGE_PATH, dataset_input_type)
